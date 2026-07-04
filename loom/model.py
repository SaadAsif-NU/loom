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
        # (1, 1, T, T) additive mask: 0 on and below the diagonal, -1e9 above.
        self._mask = np.triu(
            np.full((1, 1, config.block_size, config.block_size), _MASK_VALUE, dtype=np.float32),
            k=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, n_embd = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q = qkv[:, :, :n_embd]
        k = qkv[:, :, n_embd : 2 * n_embd]
        v = qkv[:, :, 2 * n_embd :]

        def split_heads(t: Tensor) -> Tensor:
            # (B, T, C) -> (B, H, T, C/H)
            return t.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores + Tensor(self._mask[:, :, :seq_len, :seq_len])
        weights = self.attn_dropout(F.softmax(scores, axis=-1))

        out = weights @ v  # (B, H, T, C/H)
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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
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

    def forward(
        self, ids: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``(logits, loss)``; loss is None without targets.

        The LM head reuses the token embedding matrix (weight tying), which
        both shrinks the model and improves small-scale training.
        """
        ids = np.asarray(ids)
        _, seq_len = ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        positions = np.arange(seq_len)
        x = self.drop(self.wte(ids) + self.wpe(positions))
        for block in self.blocks:
            x = block(x)
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
    ) -> np.ndarray:
        """Autoregressively extend ``ids`` (shape ``(B, T)``) by sampling.

        ``temperature`` scales the logits (0 < t; lower is greedier) and
        ``top_k`` keeps only the k most likely tokens before sampling.
        Sampling draws from the process-wide generator, so runs are
        reproducible under ``set_seed``.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive; use top_k=1 for greedy decoding")
        rng = get_rng()
        ids = np.asarray(ids)
        for _ in range(max_new_tokens):
            context = ids[:, -self.config.block_size :]
            with no_grad():
                logits, _ = self.forward(context)
            last = logits.data[:, -1, :] / temperature
            if top_k is not None:
                kth_best = np.sort(last, axis=-1)[:, -top_k][:, None]
                last = np.where(last < kth_best, -np.inf, last)
            last = last - last.max(axis=-1, keepdims=True)
            probs = np.exp(last)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_ids = np.array([rng.choice(last.shape[-1], p=row) for row in probs])
            ids = np.concatenate([ids, next_ids[:, None]], axis=1)
        return ids
