"""Behavioural tests for the autodiff engine (graph mechanics, shapes, modes)."""

import numpy as np
import pytest

from loom.tensor import Tensor, no_grad


def test_forward_values_match_numpy() -> None:
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose((a + b).data, [[11.0, 22.0], [33.0, 44.0]])
    assert np.allclose((a * b).data, [[10.0, 40.0], [90.0, 160.0]])
    assert np.allclose((a @ b).data, np.array(a.data) @ np.array(b.data))


def test_integer_input_promoted_to_float() -> None:
    t = Tensor([1, 2, 3])
    assert np.issubdtype(t.dtype, np.floating)


def test_scalar_backward_seeds_ones() -> None:
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.sum().backward()
    assert x.grad is not None
    assert np.allclose(x.grad, [1.0, 1.0, 1.0])


def test_backward_on_non_scalar_requires_gradient() -> None:
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(RuntimeError, match="scalar"):
        y.backward()


def test_backward_on_non_grad_tensor_raises() -> None:
    x = Tensor([1.0])
    with pytest.raises(RuntimeError, match="does not require grad"):
        x.sum().backward()


def test_gradients_accumulate_across_uses() -> None:
    x = Tensor([2.0], requires_grad=True)
    y = x * 3.0 + x * 4.0  # x used twice: dy/dx = 7
    y.sum().backward()
    assert x.grad is not None
    assert np.allclose(x.grad, [7.0])


def test_diamond_graph_backward_runs_once_per_node() -> None:
    # x feeds two branches that rejoin; the gradient must be the sum of both paths.
    x = Tensor([1.0, 2.0], requires_grad=True)
    a = x * 2.0
    b = x * 3.0
    ((a + b) * a).sum().backward()
    # f = (2x + 3x) * 2x = 10x^2, df/dx = 20x
    assert x.grad is not None
    assert np.allclose(x.grad, [20.0, 40.0])


def test_broadcast_add_reduces_gradient_to_bias_shape() -> None:
    x = Tensor(np.ones((4, 3)), requires_grad=True)
    bias = Tensor(np.zeros(3), requires_grad=True)
    (x + bias).sum().backward()
    assert bias.grad is not None
    assert bias.grad.shape == (3,)
    assert np.allclose(bias.grad, [4.0, 4.0, 4.0])


def test_broadcast_scalar_operand() -> None:
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    s = Tensor(3.0, requires_grad=True)
    (x * s).sum().backward()
    assert s.grad is not None
    assert s.grad.shape == ()
    assert np.allclose(s.grad, 4.0)


def test_batched_matmul_with_broadcast_weights() -> None:
    # (B, T, D) @ (D, K): the weight gradient must sum over the batch.
    x = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    w = Tensor(np.ones((4, 5)), requires_grad=True)
    (x @ w).sum().backward()
    assert w.grad is not None
    assert w.grad.shape == (4, 5)
    assert np.allclose(w.grad, np.full((4, 5), 6.0))  # 2 batches x 3 rows


def test_matmul_rejects_vectors() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        _ = Tensor([1.0, 2.0]) @ Tensor([[1.0], [2.0]])


def test_embedding_style_gather_scatter_adds_repeated_ids() -> None:
    weight = Tensor(np.zeros((5, 2)), requires_grad=True)
    ids = np.array([1, 1, 4])
    weight[ids].sum().backward()
    assert weight.grad is not None
    expected = np.zeros((5, 2))
    expected[1] = 2.0  # row 1 gathered twice
    expected[4] = 1.0
    assert np.allclose(weight.grad, expected)


def test_no_grad_suppresses_graph_recording() -> None:
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        y = x * 2.0
    assert not y.requires_grad
    assert y._backward is None


def test_no_grad_restores_state_after_exception() -> None:
    x = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="boom"), no_grad():
        raise ValueError("boom")
    assert (x * 2.0).requires_grad


def test_detach_breaks_the_graph() -> None:
    x = Tensor([1.0], requires_grad=True)
    y = (x * 2.0).detach() * 3.0
    assert not y.requires_grad


def test_zero_grad_clears_accumulated_gradient() -> None:
    x = Tensor([1.0], requires_grad=True)
    (x * 2.0).sum().backward()
    x.zero_grad()
    assert x.grad is None


def test_reverse_operators() -> None:
    x = Tensor([2.0], requires_grad=True)
    y = (1.0 - x) + (6.0 / x) + 2.0 * x + (1.0 + x)
    y.sum().backward()
    # d/dx(-x) - 6/x^2 + 2 + 1 = -1 - 1.5 + 3 = 0.5
    assert x.grad is not None
    assert np.allclose(x.grad, [0.5])


def test_swapaxes_matches_numpy() -> None:
    x = Tensor(np.arange(24.0).reshape(2, 3, 4))
    assert x.swapaxes(-1, -2).shape == (2, 4, 3)
    assert np.allclose(x.swapaxes(-1, -2).data, x.data.swapaxes(-1, -2))


def test_repr_mentions_shape_and_grad_flag() -> None:
    t = Tensor(np.zeros((2, 3)), requires_grad=True)
    assert "shape=(2, 3)" in repr(t)
    assert "requires_grad=True" in repr(t)


def test_item_and_numpy_accessors() -> None:
    t = Tensor([[42.0]])
    assert t.item() == 42.0
    assert t.numpy() is t.data
    assert t.size == 1
    assert t.ndim == 2
