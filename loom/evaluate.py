"""Held-out evaluation: mean negative log-likelihood and perplexity.

Perplexity is ``exp`` of the per-token cross-entropy over a corpus: the
effective branching factor the model still faces after training. A model
predicting uniformly over a 512-token vocabulary sits at perplexity 512;
every improvement below that is learned structure.
"""

from __future__ import annotations

import numpy as np

from loom.model import GPT
from loom.tensor import no_grad


def perplexity(model: GPT, token_ids: np.ndarray, batch_size: int = 8) -> float:
    """Perplexity of ``model`` over ``token_ids``.

    The ids are cut into non-overlapping ``block_size`` windows which are
    scored in batches (in eval mode, no dropout, no graph). The trailing
    remainder shorter than a window is dropped; with corpus-scale inputs
    that loses a negligible sliver of tokens.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    block = model.config.block_size
    num_windows = (ids.size - 1) // block
    if num_windows < 1:
        raise ValueError(f"need at least {block + 1} tokens to evaluate, got {ids.size}")

    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        total_tokens = 0
        for start in range(0, num_windows, batch_size):
            windows = range(start, min(start + batch_size, num_windows))
            x = np.stack([ids[w * block : w * block + block] for w in windows])
            y = np.stack([ids[w * block + 1 : w * block + block + 1] for w in windows])
            with no_grad():
                _, loss = model.forward(x, targets=y)
            assert loss is not None
            total_nll += loss.item() * x.size
            total_tokens += x.size
        return float(np.exp(total_nll / total_tokens))
    finally:
        model.train(was_training)
