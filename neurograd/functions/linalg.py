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
            # Match grad dtype to operand to avoid upcasting large tensors
            go = grad_output.astype(B.data.dtype, copy=False)
            grad_A = xp.matmul(go, _transpose(B.data))
        if B.requires_grad:
            # Match grad dtype to operand to avoid upcasting large tensors
            go = grad_output.astype(A.data.dtype, copy=False)
            grad_B = xp.matmul(_transpose(A.data), go)
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
        # Parse axes consistently
        if isinstance(self.axes, int):
            # Contract last n axes of A with first n axes of B
            A_axes = list(range(A.ndim - self.axes, A.ndim))
            B_axes = list(range(self.axes))
        elif isinstance(self.axes, (list, tuple)) and len(self.axes) == 2:
            A_axes, B_axes = self.axes
            # Ensure they're lists for consistency
            if isinstance(A_axes, int):
                A_axes = [A_axes]
            if isinstance(B_axes, int):
                B_axes = [B_axes]
        else:
            raise ValueError(f"Invalid axes format: {self.axes}")  
        
        # Convert negative indices to positive
        A_axes = [(ax % A.ndim) for ax in A_axes]
        B_axes = [(ax % B.ndim) for ax in B_axes]
        # Find free (non-contracted) axes
        A_free = [i for i in range(A.ndim) if i not in A_axes]
        B_free = [i for i in range(B.ndim) if i not in B_axes]
        grad_A = grad_B = None
    
        if A.requires_grad:
            # Cast incoming grad to B's dtype to prevent upcasting B (often large/strided)
            go = grad_output.astype(B.data.dtype, copy=False)
            grad_A = xp.tensordot(go, B.data,
                                axes=[list(range(len(A_free), len(A_free) + len(B_free))), B_free])
            perm = [0] * A.ndim
            for i, ax in enumerate(A_free):
                perm[ax] = i
            for i, ax in enumerate(A_axes):
                perm[ax] = len(A_free) + i
            grad_A = xp.transpose(grad_A, perm)
        
        if B.requires_grad:
            # Cast incoming grad to A's dtype to prevent upcasting A
            go = grad_output.astype(A.data.dtype, copy=False)
            grad_B = xp.tensordot(A.data, go,
                                axes=[A_free, list(range(len(A_free)))])
            perm = [0] * B.ndim
            for i, ax in enumerate(B_axes):
                perm[ax] = i
            for i, ax in enumerate(B_free):
                perm[ax] = len(B_axes) + i
            grad_B = xp.transpose(grad_B, perm)
        
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
