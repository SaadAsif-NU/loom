"""A guided tour of the loom autodiff engine.

Run from the repo root:

    python examples/autodiff_tour.py

Everything here is offline and finishes in a few seconds.
"""

import numpy as np

from loom import Tensor, set_seed
from loom import functional as F
from loom.gradcheck import gradcheck

set_seed(0)

# ----------------------------------------------------------------------
# 1. Gradients of a tiny expression
# ----------------------------------------------------------------------
print("1) d/dx of f(x) = tanh(x^2) at x = [1, 2]")
x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
f = (x**2.0).tanh()
f.sum().backward()
# Analytic: f'(x) = 2x * (1 - tanh(x^2)^2)
expected = 2 * x.data * (1 - np.tanh(x.data**2) ** 2)
print(f"   engine:   {x.grad}")
print(f"   analytic: {expected}\n")

# ----------------------------------------------------------------------
# 2. Broadcasting is handled on the way back
# ----------------------------------------------------------------------
print("2) A (4, 3) matrix plus a (3,) bias: the bias gradient sums over rows")
m = Tensor(np.ones((4, 3)), requires_grad=True)
bias = Tensor(np.zeros(3), requires_grad=True)
(m + bias).sum().backward()
print(f"   bias.grad = {bias.grad}  (each bias element fed 4 outputs)\n")

# ----------------------------------------------------------------------
# 3. The engine can be checked against finite differences at any time
# ----------------------------------------------------------------------
print("3) gradcheck: softmax gradients vs central differences")
gradcheck(lambda t: F.softmax(t, axis=-1), np.random.default_rng(0).standard_normal((3, 5)))
print("   passed\n")

# ----------------------------------------------------------------------
# 4. Train a real (tiny) network with nothing but the engine
# ----------------------------------------------------------------------
print("4) fit y = sin(x) with a 2-layer MLP, plain gradient descent")
rng = np.random.default_rng(0)
inputs = rng.uniform(-np.pi, np.pi, size=(256, 1))
targets = np.sin(inputs)

w1 = Tensor(rng.normal(0, 0.5, (1, 32)), requires_grad=True)
b1 = Tensor(np.zeros(32), requires_grad=True)
w2 = Tensor(rng.normal(0, 0.5, (32, 1)), requires_grad=True)
b2 = Tensor(np.zeros(1), requires_grad=True)
params = [w1, b1, w2, b2]

for step in range(1, 501):
    pred = F.gelu(Tensor(inputs) @ w1 + b1) @ w2 + b2
    loss = ((pred - Tensor(targets)) ** 2.0).mean()
    for p in params:
        p.zero_grad()
    loss.backward()
    for p in params:
        assert p.grad is not None
        p.data -= 0.05 * p.grad
    if step in (1, 100, 500):
        print(f"   step {step:>3}: mse = {loss.item():.5f}")

print("\nThe same Tensor class, scaled up, is what trains the GPT in loom/model.py.")
