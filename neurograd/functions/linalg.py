from neurograd import xp
from .base import Function
from neurograd.nn.module import Module

# TODO: Add more matrix operations, like splits, stacking and convolutions.

# Matrix OPS classes for Functional API
# These classes implement matrix operations like matrix multiplication, transpose, etc.
class MatMul(Function, Module):
    name = "MatMul"
    """Matrix multiplication A @ B"""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return xp.matmul(a, b)
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        grad_a = grad_b = None
        if a.requires_grad:
            # Handle 1D arrays
            if b.data.ndim == 1:
                grad_a = xp.outer(grad_output, b.data)
            else:
                grad_a = xp.matmul(grad_output, b.data.T)
        if b.requires_grad:
            # Handle 1D arrays
            if a.data.ndim == 1:
                grad_b = xp.outer(a.data, grad_output)
            else:
                grad_b = xp.matmul(a.data.T, grad_output)
        return grad_a, grad_b
    
class Transpose(Function, Module):
    name = "Transpose"
    """Transpose of a matrix"""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray) -> xp.ndarray:
        return xp.transpose(a)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        a = self.parent_tensors[0]
        return xp.swapaxes(grad_output, -2, -1) if a.requires_grad else None
    

# Convenience function for matrix multiplication
# This function is designed to be used directly with Tensor objects.
def matmul(a, b):
    return MatMul()(a, b)
def dot(a, b):
    return MatMul()(a, b)
def transpose(a):
    return Transpose()(a)