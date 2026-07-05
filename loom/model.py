"""A GPT-style decoder-only transformer built from loom primitives.

The architecture is the GPT-2 recipe at small scale: learned token and
position embeddings, pre-norm transformer blocks (causal self-attention +
GELU MLP, both with residual connections), a final layer norm, and a
language-model head tied to the token embedding matrix.

Causality comes from an additive mask: positions that would look into the
future get a large negative number added to their attention scores before
softmax, which zeroes their weight without any control flow in the graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from loom import functional as F
from loom import nn
from loom.rng import get_rng
from loom.tensor import Tensor, no_grad

_MASK_VALUE = -1e9


class LayerKV:
    """Cached keys and values for one attention layer during decoding.

    Without a cache, sampling token T+1 recomputes attention for all T
    previous tokens: O(T^2) work per generated token. The cache keeps each
    layer's keys and values (shape ``(B, H, T, head_dim)``), so each new
    token only computes its own q/k/v and attends to the stored past.
    """

    def __init__(self) -> None:
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else int(self.k.shape[2])

    def update(self, k_new: np.ndarray, v_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new keys/values along the time axis; return the full arrays."""
        if self.k is None or self.v is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = np.concatenate([self.k, k_new], axis=2)
            self.v = np.concatenate([self.v, v_new], axis=2)
        return self.k, self.v


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Set to True to capture softmax attention maps in ``last_weights``
        # on the next forward pass (visualisation/debugging only).
        self.store_weights = False
        self.last_weights: np.ndarray | None = None
        # (1, 1, T, T) additive mask: 0 on and below the diagonal, -1e9 above.
        self._mask = np.triu(
            np.full((1, 1, config.block_size, config.block_size), _MASK_VALUE, dtype=np.float32),
            k=1,
        )

    def forward(self, x: Tensor, kv_cache: LayerKV | None = None) -> Tensor:
        batch, seq_len, n_embd = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q = qkv[:, :, :n_embd]
        k = qkv[:, :, n_embd : 2 * n_embd]
        v = qkv[:, :, 2 * n_embd :]

        def split_heads(t: Tensor) -> Tensor:
            # (B, T, C) -> (B, H, T, C/H)
            return t.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # With a cache, this forward only sees the new tokens: fold the
        # stored keys/values in, and offset the causal mask so the new
        # positions may attend to the whole cached past plus themselves.
        offset = 0
        if kv_cache is not None:
            offset = kv_cache.length
            k_full, v_full = kv_cache.update(k.data, v.data)
            k, v = Tensor(k_full), Tensor(v_full)

        total_len = offset + seq_len
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores + Tensor(self._mask[:, :, offset:total_len, :total_len])
        weights = F.softmax(scores, axis=-1)
        if self.store_weights:
            self.last_weights = weights.data.copy()
        weights = self.attn_dropout(weights)

        out = weights @ v  # (B, H, T_new, C/H)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_embd)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """The transformer feed-forward block: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm residual block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, kv_cache: LayerKV | None = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """The full decoder-only language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)

        # GPT-2 trick: scale residual-path projections by 1/sqrt(2 * n_layer)
        # so activations do not grow with depth.
        scale = 1.0 / np.sqrt(2 * config.n_layer)
        for block in self.blocks:
            block.attn.proj.weight.data *= scale
            block.mlp.proj.weight.data *= scale

    def new_cache(self) -> list[LayerKV]:
        """A fresh, empty KV cache (one slot per block) for incremental decoding."""
        return [LayerKV() for _ in self.blocks]

    def forward(
        self,
        ids: np.ndarray,
        targets: np.ndarray | None = None,
        kv_cache: list[LayerKV] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``(logits, loss)``; loss is None without targets.

        With ``kv_cache``, ``ids`` holds only the tokens not yet in the
        cache; position embeddings are offset by the cache length.

        The LM head reuses the token embedding matrix (weight tying), which
        both shrinks the model and improves small-scale training.
        """
        ids = np.asarray(ids)
        _, seq_len = ids.shape
        offset = kv_cache[0].length if kv_cache else 0
        if offset + seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds block_size {self.config.block_size}"
            )

        positions = np.arange(offset, offset + seq_len)
        x = self.drop(self.wte(ids) + self.wpe(positions))
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache[i] if kv_cache else None)
        x = self.ln_f(x)

        logits = x @ self.wte.weight.swapaxes(0, 1)  # tied weights

        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(
        self,
        ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Autoregressively extend ``ids`` (shape ``(B, T)``) by sampling.

        ``temperature`` scales the logits (0 < t; lower is greedier),
        ``top_k`` keeps only the k most likely tokens, and ``top_p`` keeps
        the smallest set of tokens whose probability mass reaches p
        (nucleus sampling). Both filters may be combined.

        With ``use_cache`` (the default) the prompt is prefilled once and
        each new token attends to cached keys/values: O(T) work per token
        instead of O(T^2). Once the sequence reaches ``block_size`` the
        window slides, positions shift, and the cache is rebuilt each step,
        which degrades to the uncached cost; both paths produce identical
        samples. Sampling draws from the process-wide generator, so runs
        are reproducible under ``set_seed``.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive; use top_k=1 for greedy decoding")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        rng = get_rng()
        ids = np.asarray(ids)
        block = self.config.block_size

        cache: list[LayerKV] | None = None
        last_logits: np.ndarray | None = None
        for _ in range(max_new_tokens):
            if not use_cache or cache is None or ids.shape[1] >= block:
                # Full (re)prefill: no cache yet, caching disabled, or the
                # context window slid past block_size (positions shifted).
                context = ids[:, -block:]
                cache = self.new_cache() if use_cache and context.shape[1] < block else None
                with no_grad():
                    logits, _ = self.forward(context, kv_cache=cache)
                last_logits = logits.data[:, -1, :]
            next_ids = _sample(last_logits, temperature, top_k, top_p, rng)
            ids = np.concatenate([ids, next_ids[:, None]], axis=1)
            if cache is not None and ids.shape[1] < block:
                with no_grad():
                    logits, _ = self.forward(next_ids[:, None], kv_cache=cache)
                last_logits = logits.data[:, -1, :]
            elif cache is not None:
                cache = None  # window is full: force a re-prefill next step
        return ids


def _sample(
    logits: np.ndarray | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one token id per batch row from final-position logits."""
    assert logits is not None
    scaled = logits / temperature
    if top_k is not None:
        kth_best = np.sort(scaled, axis=-1)[:, -top_k][:, None]
        scaled = np.where(scaled < kth_best, -np.inf, scaled)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=-1, keepdims=True)
    if top_p is not None:
        # Nucleus: keep the smallest prefix of tokens (by descending
        # probability) whose cumulative mass reaches top_p.
        order = np.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, order, axis=-1)
        preceding_mass = np.cumsum(sorted_probs, axis=-1) - sorted_probs
        keep_sorted = preceding_mass < top_p  # first token always kept
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, axis=-1)
        probs = np.where(keep, probs, 0.0)
        probs /= probs.sum(axis=-1, keepdims=True)
    return np.array([rng.choice(probs.shape[-1], p=row) for row in probs])
