"""Numerical gradient checking via central differences.

This is the referee for the whole engine: every primitive op and every
composed function in loom is validated against these finite-difference
gradients in the test suite. Checks run in float64; the default tolerances
are appropriate for central differences at ``eps = 1e-6``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from loom.tensor import Tensor


def numerical_grad(
    fn: Callable[..., Tensor],
    inputs: list[np.ndarray],
    wrt: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient of ``sum(fn(*inputs) * projection)`` w.r.t. one input.

    A fixed random projection turns a tensor-valued function into a scalar
    one, which exercises every output element instead of only their sum. The
    projection is seeded identically to the one in ``gradcheck`` so the two
    sides compare the same scalar function.
    """
    rng = np.random.default_rng(0)
    output_shape = fn(*[Tensor(arr) for arr in inputs]).shape
    projection = rng.standard_normal(output_shape)

    def scalar_fn(perturbed: np.ndarray) -> float:
        args = [Tensor(perturbed if i == wrt else arr) for i, arr in enumerate(inputs)]
        return float((fn(*args).data * projection).sum())

    target = inputs[wrt]
    grad = np.zeros_like(target)
    it = np.nditer(target, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        bumped_up = target.copy()
        bumped_up[idx] += eps
        bumped_down = target.copy()
        bumped_down[idx] -= eps
        grad[idx] = (scalar_fn(bumped_up) - scalar_fn(bumped_down)) / (2 * eps)
        it.iternext()
    return grad


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs: np.ndarray,
    eps: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """Compare analytic gradients from the engine against central differences.

    Every input is treated as differentiable. Inputs must be float64 for the
    finite-difference arithmetic to have enough precision. Raises
    ``AssertionError`` with a diagnostic message on mismatch; returns True
    otherwise.
    """
    for arr in inputs:
        if arr.dtype != np.float64:
            raise TypeError("gradcheck inputs must be float64")

    tensors = [Tensor(arr, requires_grad=True) for arr in inputs]
    output = fn(*tensors)

    rng = np.random.default_rng(0)
    projection = rng.standard_normal(output.shape)
    (output * Tensor(projection)).sum().backward()

    for i, tensor in enumerate(tensors):
        numeric = numerical_grad(fn, list(inputs), wrt=i, eps=eps)
        analytic = tensor.grad
        assert analytic is not None, f"input {i}: no gradient was accumulated"
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            worst = np.abs(analytic - numeric).max()
            raise AssertionError(
                f"input {i}: analytic and numerical gradients disagree "
                f"(max abs diff {worst:.3e})\nanalytic:\n{analytic}\nnumerical:\n{numeric}"
            )
    return True
