"""Measure what the KV cache buys: tokens/sec with and without it.

Usage:

    python scripts/benchmark_generation.py

Uses a randomly initialised model at the README config (weights do not
affect speed) and greedy decoding so both paths do identical sampling work.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from loom.model import GPT, GPTConfig
from loom.rng import set_seed

CONFIG = GPTConfig(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
PROMPT_LEN = 32
NEW_TOKENS = 64


def run(model: GPT, use_cache: bool) -> float:
    prompt = np.ones((1, PROMPT_LEN), dtype=np.int64)
    model.generate(prompt, max_new_tokens=4, top_k=1, use_cache=use_cache)  # warm-up
    start = time.perf_counter()
    model.generate(prompt, max_new_tokens=NEW_TOKENS, top_k=1, use_cache=use_cache)
    return NEW_TOKENS / (time.perf_counter() - start)


def main() -> int:
    set_seed(0)
    model = GPT(CONFIG)
    model.eval()
    print(f"model: {model.num_parameters():,} params, prompt {PROMPT_LEN}, {NEW_TOKENS} new tokens")
    without = run(model, use_cache=False)
    with_cache = run(model, use_cache=True)
    print(f"without KV cache: {without:6.1f} tokens/sec")
    print(f"with KV cache:    {with_cache:6.1f} tokens/sec  ({with_cache / without:.1f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
