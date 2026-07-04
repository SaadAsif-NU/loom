"""Optimizer, clipping, and schedule tests."""

import numpy as np
import pytest

from loom import nn
from loom.optim import AdamW, ParamGroup, clip_grad_norm, cosine_lr, param_groups
from loom.tensor import Tensor


def quadratic_param(start: float = 5.0) -> Tensor:
    return Tensor(np.array([start]), requires_grad=True)


def test_adamw_converges_on_quadratic() -> None:
    # Minimise f(x) = x^2; AdamW should walk x from 5 to ~0.
    x = quadratic_param(5.0)
    opt = AdamW(groups=[ParamGroup(params=[x])], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        (x * x).sum().backward()
        opt.step()
    assert abs(float(x.data[0])) < 1e-2


def test_adamw_first_step_matches_reference() -> None:
    # With bias correction, the first Adam step is lr * g / (|g| + eps).
    x = Tensor(np.array([2.0]), requires_grad=True)
    opt = AdamW(groups=[ParamGroup(params=[x])], lr=0.5)
    opt.zero_grad()
    (x * 3.0).sum().backward()  # constant gradient of 3
    opt.step()
    expected = 2.0 - 0.5 * 3.0 / (3.0 + opt.eps)
    assert np.allclose(x.data, [expected], atol=1e-10)


def test_decoupled_weight_decay_shrinks_without_gradient_signal() -> None:
    # Zero loss gradient: pure decay should still shrink the weight,
    # and it must not pass through the Adam moments.
    x = Tensor(np.array([4.0]), requires_grad=True)
    opt = AdamW(groups=[ParamGroup(params=[x], weight_decay=0.1)], lr=0.01)
    opt.zero_grad()
    (x * 0.0).sum().backward()
    opt.step()
    assert np.allclose(x.data, [4.0 - 0.01 * 0.1 * 4.0])


def test_params_without_grad_are_skipped() -> None:
    x = Tensor(np.array([1.0]), requires_grad=True)
    opt = AdamW(groups=[ParamGroup(params=[x])], lr=0.1)
    opt.step()  # no backward ran; nothing should change or crash
    assert np.allclose(x.data, [1.0])


def test_param_groups_split_matrices_from_vectors() -> None:
    layer = nn.Linear(4, 4)
    groups = param_groups(layer, weight_decay=0.1)
    assert groups[0].weight_decay == 0.1
    assert all(p.ndim >= 2 for p in groups[0].params)  # weight matrix decays
    assert groups[1].weight_decay == 0.0
    assert all(p.ndim < 2 for p in groups[1].params)  # bias does not


def test_clip_grad_norm_scales_down_only_when_needed() -> None:
    a = Tensor(np.array([3.0]), requires_grad=True)
    b = Tensor(np.array([4.0]), requires_grad=True)
    a.grad = np.array([3.0])
    b.grad = np.array([4.0])  # global norm = 5
    returned = clip_grad_norm([a, b], max_norm=1.0)
    assert returned == pytest.approx(5.0)
    assert np.sqrt((a.grad**2 + b.grad**2).sum()) == pytest.approx(1.0, abs=1e-6)

    c = Tensor(np.array([0.3]), requires_grad=True)
    c.grad = np.array([0.3])
    clip_grad_norm([c], max_norm=1.0)
    assert np.allclose(c.grad, [0.3])  # under the limit: untouched


def test_clip_grad_norm_empty_is_zero() -> None:
    assert clip_grad_norm([Tensor(np.array([1.0]), requires_grad=True)], 1.0) == 0.0


def test_cosine_lr_shape() -> None:
    kwargs = {"max_lr": 1.0, "min_lr": 0.1, "warmup_steps": 10, "total_steps": 100}
    # Linear warmup from max_lr/warmup to max_lr.
    assert cosine_lr(0, **kwargs) == pytest.approx(0.1)
    assert cosine_lr(9, **kwargs) == pytest.approx(1.0)
    # Monotone decay afterwards.
    values = [cosine_lr(s, **kwargs) for s in range(10, 100)]
    assert all(earlier >= later for earlier, later in zip(values, values[1:], strict=False))
    # Halfway through decay: midpoint of max and min.
    assert cosine_lr(55, **kwargs) == pytest.approx(0.55, abs=0.01)
    # Floor at and beyond the end.
    assert cosine_lr(100, **kwargs) == pytest.approx(0.1)
    assert cosine_lr(10_000, **kwargs) == pytest.approx(0.1)
