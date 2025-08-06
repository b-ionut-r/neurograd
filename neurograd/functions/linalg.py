from neurograd import xp
from .base import Function
from neurograd.nn.module import Module
from typing import TYPE_CHECKING, Union, Tuple, Sequence
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from neurograd.tensor import Tensor


# Matrix OPS classes for Functional API
# These classes implement matrix operations like matrix/tensor dot products, transpose, etc.
# with axes handling
class MatMul(Function, Module):
    name = "MatMul"
    """Matrix multiplication A @ B with support for higher dimensions"""
    
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, A: xp.ndarray, B: xp.ndarray) -> xp.ndarray:
        return xp.matmul(A, B)
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        A, B = self.parent_tensors
        grad_A = grad_B = None
        def _transpose(arr):
            """Transpose that works for any ndim >= 1"""
            if arr.ndim == 1:
                return arr  # 1D arrays don't transpose
            elif arr.ndim == 2:
                return arr.T  # Use .T for 2D for efficiency
            else:
                return xp.swapaxes(arr, -2, -1)  # Swap last two axes for higher dims
        if A.requires_grad:
            # grad_A = grad_output @ B.T
            grad_A = xp.matmul(grad_output, _transpose(B.data))
        if B.requires_grad:
            # grad_B = A.T @ grad_output  
            grad_B = xp.matmul(_transpose(A.data), grad_output)
        return grad_A, grad_B


class TensorDot(Function, Module):
    name = "TensorDot"
    """Tensor contraction along specified axes"""
    def __init__(self, axes):
        Function.__init__(self)
        Module.__init__(self)
        self.axes = axes
        self.output_shape = None
    def forward(self, A: xp.ndarray, B: xp.ndarray) -> xp.ndarray:
        C = xp.tensordot(A, B, axes=self.axes)
        self.output_shape = C.shape
        return C
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A, B = self.parent_tensors
        if isinstance(self.axes, int):
            # When axes is an int n, contract last n axes of a with first n axes of b
            A_axes = list(range(A.ndim - self.axes, A.ndim))
            B_axes = list(range(self.axes))
        elif isinstance(self.axes, (list, tuple)) and len(self.axes) == 2:
            A_axes, B_axes = self.axes
        else:
            # Single axis case
            A_axes = [self.axes] if isinstance(self.axes, int) else self.axes
            B_axes = [self.axes] if isinstance(self.axes, int) else self.axes
        A_free_axes = [ax for ax in range(A.ndim) if ax not in A_axes]
        B_free_axes = [ax for ax in range(B.ndim) if ax not in B_axes]
        output_ndim = A.ndim + B.ndim - len(A_axes) - len(B_axes)
        grad_A = grad_B = None
        if A.requires_grad:
            grad_A = xp.tensordot(grad_output, B.data, 
                                  axes=[list(range(output_ndim))[-len(B_free_axes):], B_free_axes])
            order = A_axes + [ax for ax in range(A.ndim) if ax not in A_axes]
            inverse_perm = [0] * A.ndim
            for i, ax in enumerate(order): inverse_perm[ax] = i
            grad_A = xp.transpose(grad_A, inverse_perm)
        if B.requires_grad:
            grad_B = xp.tensordot(A.data, grad_output,
                                  axes=[A_free_axes, list(range(output_ndim))[:len(A_free_axes)]])
            order = B_axes + [ax for ax in range(B.ndim) if ax not in B_axes]
            inverse_perm = [0] * B.ndim
            for i, ax in enumerate(order): inverse_perm[ax] = i
            grad_B = xp.transpose(grad_B, inverse_perm)
        return grad_A, grad_B


class Transpose(Function, Module):
    name = "Transpose"
    """Transpose of a matrix"""
    def __init__(self, axes=None):
        Function.__init__(self)
        Module.__init__(self)
        self.axes = axes # tuple of permuation
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        if self.axes is None:
            self.axes = tuple(range(A.ndim - 2)) + (A.ndim - 1, A.ndim - 2)
        return xp.transpose(A, axes=self.axes)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        if not A.requires_grad:
            return None
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i
        return xp.transpose(grad_output, axes=inv_axes) # inverse axes permuation


# Convenience function for matrix multiplication
# This function is designed to be used directly with Tensor objects.
def matmul(A, B):
    return MatMul()(A, B)
def dot(A, B):
    return MatMul()(A, B)
def tensordot(A, B, axes):
    return TensorDot(axes)(A, B)
def transpose(A, axes=None):
    return Transpose(axes)(A)
