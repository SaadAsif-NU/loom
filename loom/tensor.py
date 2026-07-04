"""Reverse-mode automatic differentiation on NumPy arrays.

This is the engine everything else in loom is built on. The design is
define-by-run: every operation on a ``Tensor`` immediately computes its
result with NumPy and, if gradients are enabled, records a backward
closure plus references to its parent tensors. Calling ``backward()`` on
a scalar result topologically sorts the recorded graph and runs the
closures in reverse order, accumulating gradients into every tensor that
was created with ``requires_grad=True``.

Two details matter more than the rest and are easy to get wrong:

1. Broadcasting. NumPy silently broadcasts operands to a common shape on
   the forward pass. On the backward pass the incoming gradient has the
   broadcast shape, so it must be summed back down to each operand's
   original shape (see ``_unbroadcast``). Skipping this produces
   gradients with the wrong shape or, worse, silently wrong values.

2. Gradient accumulation. A tensor used in several places receives the
   sum of the gradients from each use, which is why closures add into
   ``grad`` rather than assign.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, TypeAlias

import numpy as np

Scalar: TypeAlias = "int | float"
TensorLike: TypeAlias = "Tensor | np.ndarray | int | float | list"

_grad_enabled: bool = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph recording inside the block (like ``torch.no_grad``)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    Broadcasting grows an operand in two ways: by prepending new leading
    axes, and by stretching existing axes of size 1. Both are undone by
    summing, because every broadcast copy contributed to the output and
    d(out)/d(operand) sums over the copies.
    """
    extra_dims = grad.ndim - len(shape)
    if extra_dims > 0:
        grad = grad.sum(axis=tuple(range(extra_dims)))
    stretched = tuple(
        i for i, (g, s) in enumerate(zip(grad.shape, shape, strict=True)) if s == 1 and g != 1
    )
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that records the graph needed for backprop."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents")

    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool = False,
        _parents: tuple[Tensor, ...] = (),
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        self.data: np.ndarray = array
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad and _grad_enabled
        self._backward: Callable[[], None] | None = None
        self._parents: tuple[Tensor, ...] = _parents

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def __repr__(self) -> str:
        grad_note = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{grad_note})"

    # ------------------------------------------------------------------
    # Graph plumbing
    # ------------------------------------------------------------------

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = grad.astype(self.data.dtype, copy=True)
        else:
            self.grad += grad

    @staticmethod
    def _make(
        data: np.ndarray,
        parents: tuple[Tensor, ...],
        backward: Callable[[np.ndarray], None],
    ) -> Tensor:
        """Build a result tensor, wiring the backward closure if any parent needs grad."""
        needs_grad = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs_grad, _parents=parents if needs_grad else ())

        if needs_grad:

            def run_backward() -> None:
                assert out.grad is not None
                backward(out.grad)

            out._backward = run_backward
        return out

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Run reverse-mode autodiff from this tensor.

        Without an explicit ``grad`` the tensor must be a scalar (the
        usual "call backward on the loss" case).
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("backward() without a gradient requires a scalar tensor")
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=self.data.dtype).reshape(self.shape).copy()

        ordered: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                ordered.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

        for node in reversed(ordered):
            if node._backward is not None:
                node._backward()

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------------
    # Arithmetic primitives
    # ------------------------------------------------------------------

    def __add__(self, other: TensorLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(grad, self.shape))
            if other_t.requires_grad:
                other_t._accumulate(_unbroadcast(grad, other_t.shape))

        return Tensor._make(self.data + other_t.data, (self, other_t), backward)

    def __mul__(self, other: TensorLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(grad * other_t.data, self.shape))
            if other_t.requires_grad:
                other_t._accumulate(_unbroadcast(grad * self.data, other_t.shape))

        return Tensor._make(self.data * other_t.data, (self, other_t), backward)

    def __pow__(self, exponent: Scalar) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("only scalar exponents are supported")

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad * exponent * self.data ** (exponent - 1))

        return Tensor._make(self.data**exponent, (self,), backward)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: TensorLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other_t)

    def __truediv__(self, other: TensorLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self * other_t**-1.0

    def __radd__(self, other: TensorLike) -> Tensor:
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        return self * other

    def __rsub__(self, other: TensorLike) -> Tensor:
        return Tensor(other) + (-self)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return Tensor(other) * self**-1.0

    def __matmul__(self, other: TensorLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim < 2 or other_t.ndim < 2:
            raise ValueError("matmul requires tensors with at least 2 dimensions")

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                grad_a = grad @ other_t.data.swapaxes(-1, -2)
                self._accumulate(_unbroadcast(grad_a, self.shape))
            if other_t.requires_grad:
                grad_b = self.data.swapaxes(-1, -2) @ grad
                other_t._accumulate(_unbroadcast(grad_b, other_t.shape))

        return Tensor._make(self.data @ other_t.data, (self, other_t), backward)

    # ------------------------------------------------------------------
    # Elementwise nonlinearities
    # ------------------------------------------------------------------

    def exp(self) -> Tensor:
        result = np.exp(self.data)

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad * result)

        return Tensor._make(result, (self,), backward)

    def log(self) -> Tensor:
        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad / self.data)

        return Tensor._make(np.log(self.data), (self,), backward)

    def tanh(self) -> Tensor:
        result = np.tanh(self.data)

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad * (1.0 - result**2))

        return Tensor._make(result, (self,), backward)

    def relu(self) -> Tensor:
        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad * (self.data > 0))

        return Tensor._make(np.maximum(self.data, 0.0), (self,), backward)

    def sqrt(self) -> Tensor:
        return self**0.5

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        def backward(grad: np.ndarray) -> None:
            if not self.requires_grad:
                return
            expanded = grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(a % self.ndim for a in axes):
                    expanded = np.expand_dims(expanded, ax)
            self._accumulate(np.broadcast_to(expanded, self.shape).copy())

        if keepdims:
            result = np.asarray(self.data.sum(axis=axis, keepdims=True))
        else:
            result = np.asarray(self.data.sum(axis=axis))
        return Tensor._make(result, (self,), backward)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[a % self.ndim] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, *shape: int) -> Tensor:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad.reshape(self.shape))

        return Tensor._make(self.data.reshape(target), (self,), backward)

    def transpose(self, *axes: int) -> Tensor:
        permutation = axes if axes else tuple(reversed(range(self.ndim)))
        inverse = tuple(int(i) for i in np.argsort(permutation))

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate(grad.transpose(inverse))

        return Tensor._make(self.data.transpose(permutation), (self,), backward)

    def swapaxes(self, axis_a: int, axis_b: int) -> Tensor:
        permutation = list(range(self.ndim))
        permutation[axis_a], permutation[axis_b] = permutation[axis_b], permutation[axis_a]
        return self.transpose(*permutation)

    def __getitem__(self, index: Any) -> Tensor:
        """Slice or gather. Integer-array indexing doubles as embedding lookup:
        ``weight[ids]`` gathers rows, and the backward pass scatter-adds into
        the weight gradient (``np.add.at`` handles repeated indices correctly).
        """

        def backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                full = np.zeros_like(self.data)
                np.add.at(full, index, grad)
                self._accumulate(full)

        return Tensor._make(self.data[index], (self,), backward)
