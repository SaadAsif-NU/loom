"""Module-system tests: parameter discovery, modes, layers, state dicts."""

import numpy as np
import pytest

from loom import nn
from loom.rng import set_seed
from loom.tensor import Tensor


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.blocks = [nn.Linear(8, 8), nn.Linear(8, 8)]
        self.norm = nn.LayerNorm(8)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.drop(self.blocks[1](self.blocks[0](self.fc1(x)))))


def test_named_parameters_produces_dotted_paths() -> None:
    net = TinyNet()
    names = [name for name, _ in net.named_parameters()]
    assert "fc1.weight" in names
    assert "blocks.0.weight" in names
    assert "blocks.1.bias" in names
    assert "norm.weight" in names
    assert len(names) == len(set(names)), "parameter names must be unique"


def test_parameters_counts_everything_trainable() -> None:
    net = TinyNet()
    # fc1: 4*8+8, two blocks: 2*(8*8+8), norm: 8+8
    assert net.num_parameters() == (4 * 8 + 8) + 2 * (8 * 8 + 8) + 16


def test_linear_without_bias_has_no_bias_parameter() -> None:
    layer = nn.Linear(3, 5, bias=False)
    assert [name for name, _ in layer.named_parameters()] == ["weight"]


def test_train_eval_propagates_through_lists() -> None:
    net = TinyNet()
    net.eval()
    assert not net.drop.training
    assert not net.blocks[0].training
    net.train()
    assert net.drop.training


def test_zero_grad_clears_all_parameters() -> None:
    net = TinyNet()
    out = net(Tensor(np.random.default_rng(0).standard_normal((2, 4))))
    out.sum().backward()
    assert any(p.grad is not None for p in net.parameters())
    net.zero_grad()
    assert all(p.grad is None for p in net.parameters())


def test_linear_matches_manual_affine() -> None:
    layer = nn.Linear(3, 2)
    x = np.random.default_rng(0).standard_normal((5, 3))
    expected = x @ layer.weight.data + layer.bias.data  # type: ignore[union-attr]
    assert np.allclose(layer(Tensor(x)).data, expected, atol=1e-6)


def test_embedding_gathers_rows() -> None:
    emb = nn.Embedding(10, 4)
    ids = np.array([[1, 3], [7, 1]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    assert np.allclose(out.data, emb.weight.data[ids])


def test_dropout_eval_is_identity() -> None:
    drop = nn.Dropout(0.9)
    drop.eval()
    x = Tensor(np.ones((100,)))
    assert np.allclose(drop(x).data, x.data)


def test_dropout_train_masks_and_rescales() -> None:
    set_seed(0)
    drop = nn.Dropout(0.25)
    x = Tensor(np.ones((10_000,)))
    out = drop(x).data
    dropped = (out == 0).mean()
    assert 0.2 < dropped < 0.3  # roughly p of the units dropped
    kept = out[out != 0]
    assert np.allclose(kept, 1.0 / 0.75)  # survivors rescaled by 1/(1-p)
    assert 0.9 < out.mean() < 1.1  # expectation preserved


def test_dropout_rejects_bad_probability() -> None:
    with pytest.raises(ValueError, match="dropout probability"):
        nn.Dropout(1.0)


def test_layernorm_module_normalises_last_axis() -> None:
    norm = nn.LayerNorm(16)
    x = Tensor(np.random.default_rng(0).standard_normal((4, 16)) * 5 + 3)
    out = norm(x).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_state_dict_round_trip() -> None:
    set_seed(1)
    net_a = TinyNet()
    set_seed(2)
    net_b = TinyNet()
    assert not np.allclose(net_a.fc1.weight.data, net_b.fc1.weight.data)
    net_b.load_state_dict(net_a.state_dict())
    for (_, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters(), strict=True):
        assert np.allclose(pa.data, pb.data)


def test_load_state_dict_rejects_missing_and_unexpected_keys() -> None:
    net = TinyNet()
    state = net.state_dict()
    state.pop("fc1.weight")
    state["bogus"] = np.zeros(3)
    with pytest.raises(ValueError, match="state dict mismatch"):
        net.load_state_dict(state)


def test_load_state_dict_rejects_shape_mismatch() -> None:
    net = TinyNet()
    state = net.state_dict()
    state["fc1.weight"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        net.load_state_dict(state)


def test_set_seed_makes_initialisation_reproducible() -> None:
    set_seed(7)
    a = nn.Linear(4, 4).weight.data.copy()
    set_seed(7)
    b = nn.Linear(4, 4).weight.data.copy()
    assert np.array_equal(a, b)
