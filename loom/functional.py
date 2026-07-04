"""Higher-level differentiable functions composed from Tensor primitives.

Nothing in this module implements its own backward pass. Each function is
written in terms of the primitive ops in ``loom.tensor``, so the engine
differentiates it automatically. The one trick used throughout is the
*detached constant*: subtracting ``max(x)`` (as a plain constant, outside
the graph) before exponentiating makes softmax and log-sum-exp numerically
stable without changing either the value or the gradient, because
``softmax(x - c) == softmax(x)`` for any constant ``c``.
"""

from __future__ import annotations

import numpy as np

from loom.tensor import Tensor

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax along ``axis``, stabilised by shifting with a detached max."""
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    exps = (x - shift).exp()
    return exps / exps.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax via the shifted log-sum-exp identity."""
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    shifted = x - shift
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def gelu(x: Tensor) -> Tensor:
    """GELU activation (tanh approximation, as used by GPT-2)."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + inner.tanh())


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy between ``logits`` and integer class ``targets``.

    ``logits`` has shape ``(..., num_classes)`` and ``targets`` the matching
    leading shape. Rows are flattened, the target log-probability is gathered
    per row, and the result is averaged. Built entirely from primitives
    (log-softmax, advanced indexing, mean), so the well-known
    ``softmax - one_hot`` gradient emerges from the engine rather than being
    hand-coded.
    """
    num_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_classes)
    flat_targets = np.asarray(targets).reshape(-1)
    if flat_targets.shape[0] != flat_logits.shape[0]:
        raise ValueError(
            f"targets shape {np.asarray(targets).shape} does not match "
            f"logits leading shape {logits.shape[:-1]}"
        )
    log_probs = log_softmax(flat_logits, axis=-1)
    rows = np.arange(flat_targets.shape[0])
    picked = log_probs[rows, flat_targets]
    return -picked.mean()


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation over the last axis with learnable scale and shift."""
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred**2).mean(axis=-1, keepdims=True)
    normed = centred / (variance + eps).sqrt()
    return normed * weight + bias
