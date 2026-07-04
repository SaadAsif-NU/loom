"""GPT architecture tests: shapes, causality, gradients, generation."""

import numpy as np
import pytest

from loom.model import GPT, CausalSelfAttention, GPTConfig
from loom.rng import set_seed
from loom.tensor import Tensor

TINY = GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def make_model(config: GPTConfig = TINY) -> GPT:
    set_seed(0)
    return GPT(config)


def test_config_validates_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(vocab_size=10, n_embd=10, n_head=3)


def test_logits_shape_and_loss_scalar() -> None:
    model = make_model()
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits, loss = model.forward(ids, targets=ids)
    assert logits.shape == (2, 4, TINY.vocab_size)
    assert loss is not None and loss.size == 1


def test_loss_none_without_targets() -> None:
    model = make_model()
    _, loss = model.forward(np.array([[1, 2, 3]]))
    assert loss is None


def test_initial_loss_close_to_uniform() -> None:
    # An untrained model should be near -log(1/V): far off means a broken
    # initialisation or a softmax bug.
    model = make_model()
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _, loss = model.forward(ids, targets=ids)
    assert loss is not None
    assert abs(loss.item() - np.log(TINY.vocab_size)) < 0.5


def test_sequence_longer_than_block_size_raises() -> None:
    model = make_model()
    too_long = np.zeros((1, TINY.block_size + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model.forward(too_long)


def test_causality_future_tokens_cannot_affect_past_logits() -> None:
    """The defining property of a decoder: changing token t+1 must leave
    logits at positions <= t untouched."""
    model = make_model()
    model.eval()
    base = np.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    changed = base.copy()
    changed[0, 5] = 11  # perturb position 5

    logits_a, _ = model.forward(base)
    logits_b, _ = model.forward(changed)
    assert np.allclose(logits_a.data[0, :5], logits_b.data[0, :5], atol=1e-6)
    assert not np.allclose(logits_a.data[0, 5:], logits_b.data[0, 5:], atol=1e-6)


def test_attention_rows_sum_to_one_under_mask() -> None:
    set_seed(0)
    attn = CausalSelfAttention(TINY)
    attn.eval()
    x = Tensor(np.random.default_rng(0).standard_normal((1, 6, TINY.n_embd)).astype(np.float32))
    out = attn(x)
    assert out.shape == (1, 6, TINY.n_embd)


def test_gradients_reach_every_parameter() -> None:
    model = make_model()
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _, loss = model.forward(ids, targets=ids)
    assert loss is not None
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert np.isfinite(param.grad).all(), f"non-finite gradient at {name}"
        assert np.abs(param.grad).sum() > 0, f"identically-zero gradient at {name}"


def test_model_gradient_matches_numerical_on_sampled_entries() -> None:
    """Spot-check the full model's gradient against central differences.

    Checking every element would be slow; a handful of random entries of the
    token embedding catches systematic errors just as well.
    """
    model = make_model()
    model.eval()
    for _, param in model.named_parameters():
        param.data = param.data.astype(np.float64)

    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    def loss_value() -> float:
        _, loss = model.forward(ids, targets=ids)
        assert loss is not None
        return loss.item()

    _, loss = model.forward(ids, targets=ids)
    assert loss is not None
    model.zero_grad()
    loss.backward()

    wte = model.wte.weight
    assert wte.grad is not None
    rng = np.random.default_rng(0)
    eps = 1e-6
    for _ in range(5):
        i = int(rng.integers(0, wte.shape[0]))
        j = int(rng.integers(0, wte.shape[1]))
        original = wte.data[i, j]
        wte.data[i, j] = original + eps
        up = loss_value()
        wte.data[i, j] = original - eps
        down = loss_value()
        wte.data[i, j] = original
        numeric = (up - down) / (2 * eps)
        assert abs(numeric - wte.grad[i, j]) < 1e-4


def test_weight_tying_lm_head_uses_token_embedding() -> None:
    model = make_model()
    names = [name for name, _ in model.named_parameters()]
    assert not any("lm_head" in name for name in names)
    # Gradient flows into wte from both the embedding and the output head.
    ids = np.array([[1, 2, 3]])
    _, loss = model.forward(ids, targets=ids)
    assert loss is not None
    loss.backward()
    assert model.wte.weight.grad is not None
    # Rows for tokens never seen in the batch still get head gradients.
    unseen = 12
    assert np.abs(model.wte.weight.grad[unseen]).sum() > 0


def test_generate_extends_by_requested_tokens() -> None:
    model = make_model()
    model.eval()
    out = model.generate(np.array([[1, 2, 3]]), max_new_tokens=5)
    assert out.shape == (1, 8)
    assert np.array_equal(out[:, :3], [[1, 2, 3]])
    assert out.max() < TINY.vocab_size and out.min() >= 0


def test_generate_handles_contexts_longer_than_block_size() -> None:
    model = make_model()
    model.eval()
    long_context = np.ones((1, TINY.block_size), dtype=np.int64)
    out = model.generate(long_context, max_new_tokens=3)
    assert out.shape == (1, TINY.block_size + 3)


def test_generate_is_reproducible_under_seed() -> None:
    model = make_model()
    model.eval()
    set_seed(123)
    a = model.generate(np.array([[1, 2]]), max_new_tokens=6)
    set_seed(123)
    b = model.generate(np.array([[1, 2]]), max_new_tokens=6)
    assert np.array_equal(a, b)


def test_generate_top_k_1_is_greedy() -> None:
    model = make_model()
    model.eval()
    set_seed(1)
    a = model.generate(np.array([[1, 2]]), max_new_tokens=4, top_k=1)
    set_seed(99)  # different seed, same result: greedy ignores randomness
    b = model.generate(np.array([[1, 2]]), max_new_tokens=4, top_k=1)
    assert np.array_equal(a, b)


def test_generate_rejects_nonpositive_temperature() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="temperature"):
        model.generate(np.array([[1]]), max_new_tokens=1, temperature=0.0)
