"""Neural-network modules on top of the autodiff engine.

``Module`` provides the plumbing every layer needs: recursive parameter
discovery (used by optimizers and checkpointing), train/eval mode
propagation (used by dropout), and state-dict save/load. Layers hold their
weights as ``Tensor`` objects with ``requires_grad=True``; anything else
(masks, config) is invisible to the optimizer by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from loom import functional as F
from loom.rng import get_rng
from loom.tensor import Tensor

_INIT_STD = 0.02  # GPT-2's initialisation scale


class Module:
    """Base class: parameter discovery, mode switching, state dicts."""

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses return a Tensor (layers) or a tuple (models)."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        """All trainable tensors reachable from this module, depth-first.

        Attribute names become dotted paths (``blocks.0.attn.qkv.weight``),
        which gives checkpoints a stable, human-readable key space.
        """
        found: list[tuple[str, Tensor]] = []
        for name, value in vars(self).items():
            path = f"{prefix}{name}"
            if isinstance(value, Tensor):
                if value.requires_grad:
                    found.append((path, value))
            elif isinstance(value, Module):
                found.extend(value.named_parameters(f"{path}."))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        found.extend(item.named_parameters(f"{path}.{i}."))
        return found

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    # ------------------------------------------------------------------
    # State dicts
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        own = dict(self.named_parameters())
        missing = own.keys() - state.keys()
        unexpected = state.keys() - own.keys()
        if missing or unexpected:
            raise ValueError(
                f"state dict mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        for name, param in own.items():
            loaded = np.asarray(state[name])
            if loaded.shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {param.shape}, got {loaded.shape}"
                )
            param.data = loaded.astype(param.data.dtype, copy=True)


class Linear(Module):
    """Affine map ``x @ weight + bias`` with GPT-2 style init."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        rng = get_rng()
        self.weight = Tensor(
            rng.normal(0.0, _INIT_STD, size=(in_features, out_features)).astype(np.float32),
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Token-id (or position-id) lookup table."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        rng = get_rng()
        self.weight = Tensor(
            rng.normal(0.0, _INIT_STD, size=(num_embeddings, embedding_dim)).astype(np.float32),
            requires_grad=True,
        )

    def forward(self, ids: np.ndarray) -> Tensor:
        return self.weight[np.asarray(ids)]


class LayerNorm(Module):
    """Layer normalisation with learnable scale and shift."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, eps=self.eps)


class Dropout(Module):
    """Inverted dropout: identity in eval mode, rescaled mask in train mode."""

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (get_rng().random(x.shape) >= self.p).astype(x.data.dtype)
        return x * Tensor(keep / (1.0 - self.p))
