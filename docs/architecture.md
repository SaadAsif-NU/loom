# loom architecture

loom rebuilds the language-model stack from primitives in four layers, each
one testable on its own and used unchanged by the layer above:

```
loom train / loom generate / loom eval        (cli.py)
        |
Trainer, checkpoints, perplexity              (train.py, evaluate.py)
        |
GPT: blocks, attention, MLP                   (model.py, nn.py)
        |
Tensor autodiff + functional ops              (tensor.py, functional.py)
        |
BPE tokenizer                                 (tokenizer.py)
```

## The autodiff engine (`tensor.py`)

The engine is define-by-run reverse-mode autodiff, the same execution model
as PyTorch eager. A `Tensor` wraps a NumPy array; every operation computes
its result immediately and, when gradients are enabled, records a closure
that knows how to push gradients back to its parents. `backward()` walks the
recorded graph once, in reverse topological order (iteratively, so deep
graphs cannot hit the recursion limit), and accumulates into `grad`.

Design decisions worth knowing:

- **Only ~17 primitives.** add, mul, pow, matmul, exp, log, tanh, relu,
  sum, mean, reshape, transpose, and indexing cover everything a
  transformer needs. Fewer primitives means fewer hand-derived gradients
  and a smaller correctness surface: everything else is composition.
- **Broadcasting is undone by summation.** NumPy broadcasts operands
  forward; the backward pass must sum the gradient over every axis that
  was created or stretched (`_unbroadcast`). This is the classic silent
  bug in from-scratch engines, which is why it has dedicated tests for
  new-axis, stretched-axis, and batched-matmul cases.
- **Indexing is the embedding op.** `weight[ids]` with an integer array
  gathers rows; its backward scatter-adds with `np.add.at`, which handles
  repeated ids correctly. No separate embedding primitive needed.
- **Gradient accumulation, not assignment.** A tensor used twice receives
  the sum of both paths (tested with a diamond-shaped graph).

## Composed functions (`functional.py`)

Softmax, log-softmax, GELU, cross-entropy, and layer norm contain no
backward code at all: they are compositions of primitives, so the engine
differentiates them. Numerical stability uses the *detached constant*
trick: `softmax(x)` first subtracts `max(x)` computed outside the graph.
Because `softmax(x - c) = softmax(x)` for any constant, the value and the
gradient are both exact, but the exponentials can no longer overflow.

Cross-entropy composes log-softmax, integer-array indexing (gather the
target log-probability per row), and mean. The famous `softmax - one_hot`
gradient is never written down; a test confirms it emerges from the engine.

## Verification (`gradcheck.py`)

Every primitive and every composed function is compared against
central-difference numerical gradients in float64. A fixed random
projection reduces tensor outputs to a scalar, which exercises every
output element rather than just their sum. Two model-level checks close
the loop: random entries of the full GPT's embedding gradient are checked
numerically, and an end-to-end training test must drive the loss on a
predictable sequence from ~3.5 to under 0.15. If any gradient anywhere
were wrong, that test cannot pass.

## The tokenizer (`tokenizer.py`)

Byte-level BPE in the GPT-2 style. Text splits into pre-tokens with a
regex (so merges never cross word boundaries and a space belongs to the
word after it), each pre-token becomes UTF-8 bytes, and training
repeatedly merges the most frequent adjacent pair. Three properties are
deliberate:

- **Lossless.** The base vocabulary is all 256 bytes, so any input
  round-trips exactly; there is no unknown token.
- **Deterministic.** Ties break toward the lowest pair ids, so the same
  corpus always yields the same tokenizer.
- **Injection-safe.** Special tokens in user text are encoded as plain
  text unless the caller passes `allowed_special=True`.

Training collapses the corpus into unique pre-tokens with counts first,
so merge statistics run over tens of thousands of distinct words rather
than hundreds of thousands of tokens.

## The model (`nn.py`, `model.py`)

`nn.Module` provides recursive parameter discovery (dotted names like
`blocks.0.attn.qkv.weight`), train/eval propagation, and strict state
dicts. On top: Linear, Embedding, LayerNorm, Dropout.

The GPT follows the GPT-2 recipe at small scale:

- learned token + position embeddings, dropout on their sum
- pre-norm blocks: `x + attn(ln(x))` then `x + mlp(ln(x))`
- attention: one fused qkv projection, heads split via reshape/transpose,
  scores scaled by `1/sqrt(head_dim)`, causality enforced by adding `-1e9`
  above the diagonal before softmax (no control flow in the graph)
- MLP: expand 4x, GELU, project back
- final layer norm, then a language-model head **tied to the token
  embedding** (the logits are `x @ wte.T`), which shrinks the model and
  helps at small scale
- residual projections scaled by `1/sqrt(2 * n_layer)` at init so
  activations do not grow with depth

Causality is not assumed: a test perturbs position t and asserts logits
at positions before t are bit-for-bit unchanged.

## Training (`optim.py`, `train.py`)

AdamW implements decoupled weight decay: the decay term bypasses the
moment estimates entirely. Parameters split GPT-2 style into a decayed
group (matrices) and an undecayed group (biases, norm gains). The LR
schedule is linear warmup then cosine decay; gradients are clipped by
global norm, and the pre-clip norm is logged since a spike there is the
first symptom of divergence.

A checkpoint is one `.npz`: `model.*` arrays, `optim.m.*`/`optim.v.*`
moments, and a JSON `meta` blob (configs, step, loss history). Resuming
restores the moments, so optimization continues on the same trajectory
instead of restarting Adam cold; a test proves a resumed run keeps
learning. `model.npz` (weights only, about a third of the size) is the
shippable artifact.

## Reproducibility

One process-wide seeded generator (`rng.py`) drives initialisation,
dropout, batch sampling, and generation. `set_seed(n)` at the start of a
run makes training and sampling deterministic. The corpus is committed to
the repo; no network access is needed anywhere.

## Performance notes and non-goals

Training a ~900k-parameter model on ~575k BPE tokens runs at a few
hundred ms per step on a laptop: NumPy's BLAS does the matmuls, and the
engine adds Python overhead per op rather than per element. Deliberate
non-goals, in order of what would matter next if this were a production
system: a KV cache for generation (currently O(T^2) per sampled token),
fused attention kernels, mixed precision, and data/model parallelism.
The point of loom is that the 1,300 lines here are the honest core of
what those systems optimise.
