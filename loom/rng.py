"""One process-wide random generator so runs are reproducible end to end.

Parameter initialisation, dropout masks, batch sampling, and generation all
draw from this generator. Calling ``set_seed`` once at program start makes an
entire training run deterministic.
"""

from __future__ import annotations

import numpy as np

_rng: np.random.Generator = np.random.default_rng(0)


def set_seed(seed: int) -> None:
    """Reset the global generator. Call before building a model or training."""
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng
