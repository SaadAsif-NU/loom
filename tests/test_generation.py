"""KV-cache correctness and sampling-strategy tests.

The cache is an optimization, so the whole suite is one idea: cached and
uncached decoding must be indistinguishable except in speed.
"""

import numpy as np
import pytest

from loom.model import GPT, GPTConfig, _sample
from loom.rng import set_seed
from loom.tensor import no_grad

TINY = GPTConfig(vocab_size=23, block_size=12, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def make_model() -> GPT:
    set_seed(0)
    model = GPT(TINY)
    model.eval()
    return model


# ----------------------------------------------------------------------
# KV cache equivalence
# ----------------------------------------------------------------------


def test_incremental_logits_match_full_forward() -> None:
    """Prefill + one cached step must give the same final logits as a full
    forward over the whole sequence."""
    model = make_model()
    ids = np.array([[3, 1, 4, 1, 5, 9]])

    with no_grad():
        full_logits, _ = model.forward(ids)

    cache = model.new_cache()
    with no_grad():
        model.forward(ids[:, :-1], kv_cache=cache)  # prefill all but last
        step_logits, _ = model.forward(ids[:, -1:], kv_cache=cache)

    assert np.allclose(full_logits.data[:, -1, :], step_logits.data[:, -1, :], atol=1e-4)


def test_cache_accumulates_length_per_layer() -> None:
    model = make_model()
    cache = model.new_cache()
    with no_grad():
        model.forward(np.array([[1, 2, 3]]), kv_cache=cache)
        model.forward(np.array([[4]]), kv_cache=cache)
    assert len(cache) == TINY.n_layer
    assert all(layer.length == 4 for layer in cache)
    assert cache[0].k is not None
    assert cache[0].k.shape == (1, TINY.n_head, 4, TINY.n_embd // TINY.n_head)


def test_cache_overflow_raises() -> None:
    model = make_model()
    cache = model.new_cache()
    with no_grad():
        model.forward(np.ones((1, TINY.block_size), dtype=np.int64), kv_cache=cache)
        with pytest.raises(ValueError, match="exceeds block_size"):
            model.forward(np.array([[1]]), kv_cache=cache)


def test_greedy_generation_identical_with_and_without_cache() -> None:
    model = make_model()
    prompt = np.array([[1, 2, 3]])
    cached = model.generate(prompt, max_new_tokens=7, top_k=1, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=7, top_k=1, use_cache=False)
    assert np.array_equal(cached, uncached)


def test_sampled_generation_identical_with_and_without_cache() -> None:
    model = make_model()
    prompt = np.array([[1, 2]])
    set_seed(11)
    cached = model.generate(prompt, max_new_tokens=6, temperature=0.9, use_cache=True)
    set_seed(11)
    uncached = model.generate(prompt, max_new_tokens=6, temperature=0.9, use_cache=False)
    assert np.array_equal(cached, uncached)


def test_cached_generation_past_block_size_stays_valid() -> None:
    """Crossing block_size forces the sliding-window fallback; output must
    keep growing and stay in vocabulary."""
    model = make_model()
    prompt = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    new_tokens = TINY.block_size  # guarantees the window slides mid-run
    out = model.generate(prompt, max_new_tokens=new_tokens, use_cache=True)
    assert out.shape == (1, 8 + new_tokens)
    assert out.min() >= 0 and out.max() < TINY.vocab_size


def test_greedy_generation_across_block_boundary_matches_uncached() -> None:
    model = make_model()
    prompt = np.array([[2, 4, 6, 8, 10, 1, 3, 5, 7, 9]])  # 10 of block 12
    cached = model.generate(prompt, max_new_tokens=6, top_k=1, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=6, top_k=1, use_cache=False)
    assert np.array_equal(cached, uncached)


# ----------------------------------------------------------------------
# Nucleus (top-p) sampling
# ----------------------------------------------------------------------


def test_top_p_tiny_is_greedy() -> None:
    model = make_model()
    prompt = np.array([[1, 2]])
    set_seed(5)
    a = model.generate(prompt, max_new_tokens=5, top_p=1e-9)
    greedy = model.generate(prompt, max_new_tokens=5, top_k=1)
    assert np.array_equal(a, greedy)


def test_top_p_one_matches_unfiltered() -> None:
    model = make_model()
    prompt = np.array([[1, 2]])
    set_seed(9)
    with_p = model.generate(prompt, max_new_tokens=6, top_p=1.0)
    set_seed(9)
    without = model.generate(prompt, max_new_tokens=6)
    assert np.array_equal(with_p, without)


def test_top_p_validation() -> None:
    model = make_model()
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="top_p"):
            model.generate(np.array([[1]]), max_new_tokens=1, top_p=bad)


def test_sample_nucleus_zeroes_tail_mass() -> None:
    # Distribution [0.5, 0.3, 0.15, 0.05] with p=0.6: keep {0.5, 0.3}.
    logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    rng = np.random.default_rng(0)
    draws = {int(_sample(logits, 1.0, None, 0.6, rng)[0]) for _ in range(200)}
    assert draws <= {0, 1}
    assert draws == {0, 1}  # renormalised, both survivors actually sampled


def test_sample_combines_top_k_and_top_p() -> None:
    logits = np.log(np.array([[0.4, 0.3, 0.2, 0.1]]))
    rng = np.random.default_rng(0)
    # top_k=3 removes id 3; top_p=0.5 then keeps only id 0 (0.4/0.9 > 0.5 cum).
    draws = {int(_sample(logits, 1.0, 3, 0.4, rng)[0]) for _ in range(50)}
    assert draws == {0}


# ----------------------------------------------------------------------
# Attention-map capture
# ----------------------------------------------------------------------


def test_store_weights_captures_causal_attention_maps() -> None:
    model = make_model()
    attn = model.blocks[0].attn
    attn.store_weights = True
    with no_grad():
        model.forward(np.array([[1, 2, 3, 4, 5]]))
    maps = attn.last_weights
    assert maps is not None
    assert maps.shape == (1, TINY.n_head, 5, 5)
    assert np.allclose(maps.sum(axis=-1), 1.0, atol=1e-5)  # rows are distributions
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    for head in range(TINY.n_head):
        assert np.all(maps[0, head][upper] < 1e-6)  # no attention to the future


def test_store_weights_off_by_default() -> None:
    model = make_model()
    with no_grad():
        model.forward(np.array([[1, 2, 3]]))
    assert model.blocks[0].attn.last_weights is None
