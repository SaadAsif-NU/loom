# loom

**A small language model woven from scratch.** Reverse-mode autodiff, a byte-level BPE tokenizer, and a GPT-style transformer, all in pure Python + NumPy. No PyTorch. No TensorFlow. Every gradient is derived by hand and verified against numerical differentiation.

[![CI](https://github.com/SaadAsif-NU/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/SaadAsif-NU/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/SaadAsif-NU/loom)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## The idea

Type `loom train --data data/shakespeare.txt --out run/` and a language
model gets built in front of you: a tokenizer learns its vocabulary from
raw bytes, a transformer trains by gradient descent, and `loom generate`
speaks in the style of the corpus. The point is that **every layer of the
stack that makes this possible lives in this repo**, small enough to read
in an afternoon, tested strictly enough to trust:

| The job | What you'd normally import | What loom implements instead |
|---|---|---|
| Tokenization | tiktoken / sentencepiece | `tokenizer.py`: trainable byte-level BPE |
| Autograd | torch.autograd | `tensor.py`: reverse-mode autodiff on NumPy |
| Layers | torch.nn | `nn.py`: Module, Linear, Embedding, LayerNorm, Dropout |
| The model | transformers | `model.py`: GPT-style decoder, causal self-attention |
| Optimization | torch.optim | `optim.py`: AdamW, clipping, warmup + cosine |
| Training | a Trainer framework | `train.py`: batching, eval, resumable checkpoints |
| Fast inference | a serving engine's KV cache | `model.py`: cached O(T)-per-token decoding |

Frameworks hide the interesting parts: how gradients flow, why attention
works, what a tokenizer actually learns, what a KV cache actually caches.
loom removes the frameworks and rebuilds the stack from primitives:

- **`loom.tensor`**: a define-by-run reverse-mode autodiff engine. Tensors track their computation graph; `backward()` topologically sorts it and propagates gradients, with full NumPy broadcasting handled correctly on the way back.
- **`loom.functional`**: softmax, GELU, layer norm, and cross-entropy built by *composing* the primitive ops, so their gradients come from the engine rather than hand-derived special cases. Numerical stability (log-sum-exp shifts) is handled with detached constants.
- **`loom.tokenizer`**: a byte-level BPE tokenizer trained from raw text, GPT-2-style pre-tokenization, deterministic merges, JSON save/load, and lossless round-tripping of any UTF-8 input.
- **`loom.nn` + `loom.model`**: linear/embedding/layer-norm modules and a GPT-style decoder-only transformer with multi-head causal self-attention, **KV-cached decoding** (O(T) per generated token, proven equivalent to the uncached path by test), and temperature / top-k / nucleus sampling.
- **`loom.train`**: AdamW, gradient clipping, warmup + cosine LR schedule, gradient accumulation, batching, resumable checkpointing.

## How it is verified

"From scratch" is only interesting if it is *correct*, so the test suite
(155 tests, CI on Python 3.10 through 3.13, strict mypy, 85% coverage
gate) is built as a chain of proofs:

1. **Every primitive op** is checked against central-difference numerical
   gradients in float64, using a random projection so every output
   element is exercised.
2. **Every composed function** passes the same check, and cross-entropy's
   gradient is additionally shown to equal the textbook
   `softmax - one_hot` without ever hand-coding it.
3. **The model is provably causal**: perturb the token at position t and
   logits before t are bit-for-bit unchanged.
4. **KV-cached generation is provably faithful**: cached and uncached
   decoding produce identical outputs, including across the sliding-window
   boundary.
5. **The whole stack must learn**: an end-to-end gate trains on a
   predictable sequence and fails unless the loss falls from ~3.5 to
   under 0.15. A wrong gradient anywhere in the engine breaks this test.

## Quickstart

```bash
git clone https://github.com/SaadAsif-NU/loom.git
cd loom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Autodiff in five lines:

```python
from loom.tensor import Tensor

w = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x = Tensor([[0.5], [-1.0]])
loss = (w @ x).sum()
loss.backward()
print(w.grad)  # dloss/dw, computed by the graph
```

Train a BPE tokenizer on your own text:

```python
from loom.tokenizer import BPETokenizer

tok = BPETokenizer.train(open("data/shakespeare.txt").read(), vocab_size=512)
ids = tok.encode("To be, or not to be")
assert tok.decode(ids) == "To be, or not to be"
tok.save("tokenizer.json")
```

## Roadmap

- [x] **Autodiff engine**: broadcasting-aware Tensor with backward for add/mul/matmul/exp/log/tanh/relu/pow/sum/mean/reshape/transpose/indexing, verified by numerical gradient checks
- [x] **Composable functional layer**: softmax, log-softmax, GELU, cross-entropy built from primitives with stable log-sum-exp shifts
- [x] **Byte-level BPE tokenizer**: trainable, deterministic, lossless UTF-8 round-trip, JSON save/load
- [x] **Transformer**: embeddings, multi-head causal attention, GELU MLP blocks, pre-norm residuals, weight tying, causality verified by test
- [x] **Training loop**: AdamW (decoupled decay, GPT-2 style param groups), grad clipping, warmup + cosine schedule, resumable checkpoints with optimizer state
- [x] **Sampling**: temperature, top-k, and nucleus (top-p) sampling with KV-cached decoding, equivalence-tested against the uncached path
- [x] **CLI + docs**: `loom train` / `loom generate` / `loom eval`, architecture writeup, runnable examples
- [x] **Training dashboard**: live loss curves, real-time metrics, model generation panel, FastAPI + WebSocket
- [ ] **Trained model**: full Shakespeare run with loss curves, perplexity eval, committed checkpoint

## CLI

The whole workflow is available as `loom` commands (also via `python -m loom`):

```bash
# Launch the interactive training dashboard (http://127.0.0.1:8000)
loom serve

# Or train from the command line
loom train --data data/shakespeare.txt --out checkpoints/shakespeare \
    --vocab-size 512 --block-size 128 --n-layer 4 --n-head 4 --n-embd 128 \
    --steps 2000 --batch-size 32 --lr 1e-3

# Sample from the trained model (temperature / top-k / top-p)
loom generate --checkpoint checkpoints/shakespeare/model.npz \
    --tokenizer checkpoints/shakespeare/tokenizer.json \
    --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top-k 40

# Held-out perplexity
loom eval --checkpoint checkpoints/shakespeare/model.npz \
    --tokenizer checkpoints/shakespeare/tokenizer.json \
    --data data/shakespeare.txt
```

Interrupted runs restart exactly where they left off (`--resume` reloads
the weights, the Adam moments, and the step counter from `checkpoint.npz`).

## Training Dashboard

Run `loom serve` to launch an interactive training dashboard at `http://127.0.0.1:8000`. The dashboard provides:

- **Real-time loss curves** with live updates as training progresses
- **Live metrics**: step count, loss, validation loss, learning rate, ETA
- **Model generation panel**: sample text from the trained model with temperature / top-k control
- **Flexible configuration**: adjust hyperparameters, model architecture, batch size, learning rate all from the UI
- **Resume training**: pick up from a checkpoint and continue training
- **Simple defaults**: click Start Training and the dashboard runs on public-domain Shakespeare text

The dashboard uses FastAPI and WebSocket for zero-latency event streaming. Training runs in a background thread, freeing the UI to remain responsive. All configuration is validated server-side.

## Design notes

- **Gradients are the product.** The engine is micrograd-style define-by-run, but tensor-valued: every op records a backward closure, and `backward()` walks the graph in reverse topological order. Broadcasting is undone on the backward pass by summing over broadcast axes, which is where most from-scratch engines quietly get gradients wrong.
- **Compose, do not special-case.** Softmax and cross-entropy are written in terms of primitive ops plus detached constants for stability, so the engine differentiates them automatically. If the primitives are right (and they are gradient-checked), everything built on them is right.
- **Offline and reproducible.** The training corpus is public-domain text committed to the repo. Every run is seeded. No network, no API keys, no downloads.

## Project layout

```
loom/
  tensor.py           # autodiff engine: Tensor, primitive ops, backward()
  functional.py       # softmax, gelu, cross_entropy composed from primitives
  gradcheck.py        # central-difference numerical gradient checking
  tokenizer.py        # byte-level BPE: train / encode / decode / save / load
  nn.py               # Module base + Linear, Embedding, LayerNorm, Dropout
  model.py            # GPTConfig, causal self-attention, transformer blocks, GPT
  optim.py            # AdamW, gradient clipping, warmup + cosine LR
  train.py            # Trainer: batching, grad accumulation, resumable checkpoints
  evaluate.py         # held-out perplexity
  cli.py              # loom train / generate / eval / serve
  rng.py              # one seeded generator for init, dropout, batching, sampling
  server/
    app.py            # FastAPI app, WebSocket training event streaming
    schemas.py        # Pydantic request/response models
    static/
      index.html      # dashboard frontend
      styles.css      # responsive dark/light theme
      app.js          # live loss curve, WebSocket listener
tests/                # gradient checks, causality proof, cache equivalence, overfit gate
scripts/              # dependency-free SVG plotters + generation benchmark
data/                 # public-domain training corpus
```

Train a tiny GPT end to end (this is real code, not pseudocode):

```python
import numpy as np
from loom import GPT, GPTConfig, BPETokenizer, set_seed
from loom.train import Trainer, TrainConfig

set_seed(42)
text = open("data/shakespeare.txt").read()
tok = BPETokenizer.train(text, vocab_size=512)
ids = np.array(tok.encode(text))

model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=96))
Trainer(model=model, token_ids=ids, config=TrainConfig(max_steps=500)).train()

model.eval()
out = model.generate(np.array([tok.encode("ROMEO:")]), max_new_tokens=100, top_k=40)
print(tok.decode(list(out[0])))
```

## License

MIT
