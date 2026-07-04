"""Optimization: AdamW, global-norm gradient clipping, warmup + cosine LR.

AdamW here is the decoupled-weight-decay variant (Loshchilov & Hutter):
decay is applied directly to the weights, scaled by the learning rate, and
never enters the moment estimates. Following GPT-2 practice, only matrices
(ndim >= 2) are decayed; biases, layer-norm gains, and other vectors are
not, which ``param_groups`` encodes as two groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loom.nn import Module
from loom.tensor import Tensor


@dataclass
class ParamGroup:
    params: list[Tensor]
    weight_decay: float = 0.0


def param_groups(model: Module, weight_decay: float) -> list[ParamGroup]:
    """Split parameters GPT-2 style: matrices decay, vectors do not."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [
        ParamGroup(params=decay, weight_decay=weight_decay),
        ParamGroup(params=no_decay, weight_decay=0.0),
    ]


@dataclass
class AdamW:
    """Decoupled-weight-decay Adam with bias correction."""

    groups: list[ParamGroup]
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    _step_count: int = field(default=0, init=False)
    _m: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    _v: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    @classmethod
    def for_model(cls, model: Module, lr: float = 3e-4, weight_decay: float = 0.1) -> AdamW:
        return cls(groups=param_groups(model, weight_decay), lr=lr)

    def step(self) -> None:
        """Apply one update using the gradients currently on the parameters."""
        self._step_count += 1
        beta1, beta2 = self.betas
        bias1 = 1.0 - beta1**self._step_count
        bias2 = 1.0 - beta2**self._step_count

        for group in self.groups:
            for param in group.params:
                if param.grad is None:
                    continue
                grad = param.grad
                key = id(param)
                if key not in self._m:
                    self._m[key] = np.zeros_like(param.data)
                    self._v[key] = np.zeros_like(param.data)
                self._m[key] = beta1 * self._m[key] + (1.0 - beta1) * grad
                self._v[key] = beta2 * self._v[key] + (1.0 - beta2) * grad**2
                m_hat = self._m[key] / bias1
                v_hat = self._v[key] / bias2
                update = m_hat / (np.sqrt(v_hat) + self.eps)
                if group.weight_decay:
                    update = update + group.weight_decay * param.data
                param.data = param.data - self.lr * update

    def zero_grad(self) -> None:
        for group in self.groups:
            for param in group.params:
                param.zero_grad()


def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
    """Scale all gradients so their global L2 norm is at most ``max_norm``.

    Returns the pre-clip norm, which is worth logging: a spiking gradient
    norm is the usual first symptom of a diverging run.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = float(np.sqrt(sum(float((g**2).sum()) for g in grads)))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            g *= scale
    return total


def cosine_lr(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Linear warmup to ``max_lr``, then cosine decay to ``min_lr``.

    Steps are 0-indexed; from ``total_steps`` onward the schedule stays at
    ``min_lr``.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return float(min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))
