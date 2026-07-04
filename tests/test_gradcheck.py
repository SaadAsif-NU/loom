"""Every op and composed function, checked against numerical gradients.

These tests are the correctness contract of the engine. Inputs are float64
(central differences need the precision) and are kept away from
non-differentiable points (e.g. relu at exactly 0).
"""

import numpy as np
import pytest

from loom import functional as F
from loom.gradcheck import gradcheck
from loom.tensor import Tensor

rng = np.random.default_rng(42)


def randn(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape)


# ----------------------------------------------------------------------
# Arithmetic primitives
# ----------------------------------------------------------------------


def test_add_same_shape() -> None:
    gradcheck(lambda a, b: a + b, randn(3, 4), randn(3, 4))


def test_add_broadcast_bias() -> None:
    gradcheck(lambda a, b: a + b, randn(2, 3, 4), randn(4))


def test_add_broadcast_new_axis() -> None:
    gradcheck(lambda a, b: a + b, randn(3, 1, 4), randn(2, 1))


def test_mul_broadcast() -> None:
    gradcheck(lambda a, b: a * b, randn(2, 3), randn(3))


def test_sub_and_div() -> None:
    b = np.abs(randn(2, 3)) + 1.0  # keep the divisor away from zero
    gradcheck(lambda x, y: (x - y) / y, randn(2, 3), b)


def test_pow_scalar_exponents() -> None:
    base = np.abs(randn(3, 3)) + 0.5
    gradcheck(lambda x: x**3.0, base)
    gradcheck(lambda x: x**-0.5, base)


def test_matmul_2d() -> None:
    gradcheck(lambda a, b: a @ b, randn(3, 4), randn(4, 5))


def test_matmul_batched() -> None:
    gradcheck(lambda a, b: a @ b, randn(2, 3, 4), randn(2, 4, 5))


def test_matmul_broadcast_weights_over_batch() -> None:
    gradcheck(lambda a, b: a @ b, randn(2, 3, 4), randn(4, 5))


# ----------------------------------------------------------------------
# Nonlinearities
# ----------------------------------------------------------------------


def test_exp() -> None:
    gradcheck(lambda x: x.exp(), randn(3, 3))


def test_log() -> None:
    gradcheck(lambda x: x.log(), np.abs(randn(3, 3)) + 0.5)


def test_tanh() -> None:
    gradcheck(lambda x: x.tanh(), randn(3, 3))


def test_relu_away_from_kink() -> None:
    x = randn(4, 4)
    x[np.abs(x) < 0.1] = 0.5  # avoid the non-differentiable point at 0
    gradcheck(lambda t: t.relu(), x)


def test_sqrt() -> None:
    gradcheck(lambda x: x.sqrt(), np.abs(randn(3, 3)) + 0.5)


# ----------------------------------------------------------------------
# Reductions and shape ops
# ----------------------------------------------------------------------


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
def test_sum_axes(axis: object) -> None:
    gradcheck(lambda x: x.sum(axis=axis), randn(2, 3, 4))  # type: ignore[arg-type]


def test_sum_keepdims() -> None:
    gradcheck(lambda x: x.sum(axis=1, keepdims=True), randn(2, 3, 4))


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_mean_axes(axis: object) -> None:
    gradcheck(lambda x: x.mean(axis=axis), randn(2, 3, 4))  # type: ignore[arg-type]


def test_reshape() -> None:
    gradcheck(lambda x: x.reshape(6, 2), randn(3, 4))


def test_transpose_permutation() -> None:
    gradcheck(lambda x: x.transpose(2, 0, 1), randn(2, 3, 4))


def test_swapaxes_last_two() -> None:
    gradcheck(lambda x: x.swapaxes(-1, -2), randn(2, 3, 4))


def test_getitem_slice() -> None:
    gradcheck(lambda x: x[1:3, ::2], randn(4, 6))


def test_getitem_integer_array_gather() -> None:
    ids = np.array([0, 2, 2, 1])
    gradcheck(lambda w: w[ids], randn(4, 5))


# ----------------------------------------------------------------------
# Composed functions (loom.functional)
# ----------------------------------------------------------------------


def test_softmax() -> None:
    gradcheck(lambda x: F.softmax(x, axis=-1), randn(3, 5))


def test_softmax_large_logits_stay_stable() -> None:
    x = Tensor(np.array([[1000.0, 1000.0, 1000.0]]))
    out = F.softmax(x)
    assert np.allclose(out.data, [[1 / 3, 1 / 3, 1 / 3]])


def test_log_softmax() -> None:
    gradcheck(lambda x: F.log_softmax(x, axis=-1), randn(3, 5))


def test_gelu() -> None:
    gradcheck(F.gelu, randn(3, 4))


def test_cross_entropy_gradient() -> None:
    targets = np.array([1, 0, 3])
    gradcheck(lambda logits: F.cross_entropy(logits, targets), randn(3, 4))


def test_cross_entropy_matches_softmax_minus_onehot() -> None:
    logits = Tensor(randn(3, 4), requires_grad=True)
    targets = np.array([1, 0, 3])
    F.cross_entropy(logits, targets).backward()

    probs = F.softmax(Tensor(logits.data)).data
    onehot = np.eye(4)[targets]
    assert logits.grad is not None
    assert np.allclose(logits.grad, (probs - onehot) / 3, atol=1e-8)


def test_cross_entropy_accepts_batched_sequence_logits() -> None:
    logits = Tensor(randn(2, 3, 5), requires_grad=True)
    targets = np.array([[0, 1, 2], [3, 4, 0]])
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert loss.size == 1
    assert logits.grad is not None
    assert logits.grad.shape == (2, 3, 5)


def test_cross_entropy_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        F.cross_entropy(Tensor(randn(3, 4)), np.array([0, 1]))


def test_layer_norm_gradcheck() -> None:
    gradcheck(
        lambda x, w, b: F.layer_norm(x, w, b),
        randn(2, 3, 8),
        randn(8),
        randn(8),
    )


def test_layer_norm_normalises() -> None:
    x = Tensor(randn(4, 16))
    out = F.layer_norm(x, Tensor(np.ones(16)), Tensor(np.zeros(16)))
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.data.std(axis=-1), 1.0, atol=1e-3)


def test_mlp_end_to_end_gradcheck() -> None:
    """A two-layer MLP with GELU: the integration test for composition."""

    def mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
        return F.gelu(x @ w1 + b1) @ w2 + b2

    gradcheck(mlp, randn(4, 6), randn(6, 8), randn(8), randn(8, 3), randn(3))
