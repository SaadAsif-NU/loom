# Changelog

## 0.1.0

Initial release: the full stack, built over three days.

- **Autodiff engine**: define-by-run reverse-mode `Tensor` on NumPy with
  broadcasting-aware backward, verified against central-difference
  numerical gradients for every primitive.
- **Functional layer**: softmax, log-softmax, GELU, cross-entropy, and
  layer norm composed from primitives (no hand-written backward passes).
- **Tokenizer**: trainable byte-level BPE with deterministic merges,
  lossless UTF-8 round-tripping, injection-safe special tokens, and JSON
  persistence.
- **Model**: GPT-style decoder (pre-norm blocks, multi-head causal
  self-attention, GELU MLP, weight-tied LM head) with a causality test
  and model-level gradient spot checks.
- **Training**: AdamW with decoupled decay and GPT-2 style param groups,
  global-norm clipping, warmup + cosine schedule, resumable single-file
  checkpoints that include optimizer state.
- **Evaluation**: held-out perplexity.
- **CLI**: `loom train`, `loom generate`, `loom eval`.
- **Artifacts**: trained tiny-Shakespeare model (875k parameters)
  committed with loss curves and reproduction instructions.
