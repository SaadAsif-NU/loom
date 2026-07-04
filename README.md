# loom

**A small language model woven from scratch.** Reverse-mode autodiff, a byte-level BPE tokenizer, and a GPT-style transformer, all in pure Python + NumPy. No PyTorch. No TensorFlow. Every gradient in this repo is derived, implemented, and verified by hand.

[![CI](https://github.com/SaadAsif-NU/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/SaadAsif-NU/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/SaadAsif-NU/loom)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why this exists

Modern ML work happens on top of frameworks that hide the interesting parts: how gradients flow, why attention works, what a tokenizer actually learns. loom removes the frameworks and rebuilds the stack from primitives:

- **`loom.tensor`**: a define-by-run reverse-mode autodiff engine. Tensors track their computation graph; `backward()` topologically sorts it and propagates gradients, with full NumPy broadcasting handled correctly on the way back.
- **`loom.functional`**: softmax, GELU, layer norm, and cross-entropy built by *composing* the primitive ops, so their gradients come from the engine rather than hand-derived special cases. Numerical stability (log-sum-exp shifts) is handled with detached constants.
- **`loom.tokenizer`**: a byte-level BPE tokenizer trained from raw text, GPT-2-style pre-tokenization, deterministic merges, JSON save/load, and lossless round-tripping of any UTF-8 input.
- **`loom.nn` + `loom.model`**: linear/embedding/layer-norm modules and a GPT-style decoder-only transformer with multi-head causal self-attention.
- **`loom.train`**: AdamW, gradient clipping, warmup + cosine LR schedule, batching, checkpointing.

Everything is verified: each primitive op and every composed function is checked against central-difference numerical gradients in the test suite.

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
- [x] **Sampling**: autoregressive generation with temperature and top-k
- [ ] **Trained model**: full Shakespeare run with loss curves, perplexity eval, committed checkpoint
- [ ] **CLI + docs**: `loom train` / `loom generate`, architecture writeup

## Design notes

- **Gradients are the product.** The engine is micrograd-style define-by-run, but tensor-valued: every op records a backward closure, and `backward()` walks the graph in reverse topological order. Broadcasting is undone on the backward pass by summing over broadcast axes, which is where most from-scratch engines quietly get gradients wrong.
- **Compose, do not special-case.** Softmax and cross-entropy are written in terms of primitive ops plus detached constants for stability, so the engine differentiates them automatically. If the primitives are right (and they are gradient-checked), everything built on them is right.
- **Offline and reproducible.** The training corpus is public-domain text committed to the repo. Every run is seeded. No network, no API keys, no downloads.

## Project layout

```
loom/
  tensor.py       # autodiff engine: Tensor, primitive ops, backward()
  functional.py   # softmax, gelu, cross_entropy composed from primitives
  gradcheck.py    # central-difference numerical gradient checking
  tokenizer.py    # byte-level BPE: train / encode / decode / save / load
  nn.py           # Module base + Linear, Embedding, LayerNorm, Dropout
  model.py        # GPTConfig, causal self-attention, transformer blocks, GPT
  optim.py        # AdamW, gradient clipping, warmup + cosine LR
  train.py        # Trainer: batching, eval, resumable checkpoints
  rng.py          # one seeded generator for init, dropout, batching, sampling
tests/            # gradient checks, causality proof, overfit sanity check
data/             # public-domain training corpus
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
