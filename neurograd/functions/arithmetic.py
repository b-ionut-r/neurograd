import neurograd as ng
from neurograd import xp
from .base import Function
from neurograd.nn.module import Module

### Element-wise operations classes for Functional API
class Add(Function, Module):
    name = "Add"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a + b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        a_grad = self._handle_broadcasting(grad_output, a.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(grad_output, b.shape) if b.requires_grad else None
        return a_grad, b_grad

class Sub(Function, Module):
    name = "Sub"    
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a - b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        a_grad = self._handle_broadcasting(grad_output, a.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(-grad_output, b.shape) if b.requires_grad else None
        return a_grad, b_grad

class Mul(Function, Module):
    """Element-wise multiplication."""
    name = "Mul"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a * b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        a_grad = self._handle_broadcasting(grad_output * b.data, a.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(grad_output * a.data, b.shape) if b.requires_grad else None
        return a_grad, b_grad

class Div(Function, Module):
    """Element-wise division."""
    name = "Div"
    @ng.fuse
    def _fused_bw_b(grad_output, a, b):
        return -grad_output * a / (b ** 2)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a / b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        a_grad = grad_output / b.data if a.requires_grad else None
        b_grad = Div._fused_bw_b(grad_output, a.data, b.data) if b.requires_grad else None
        return (self._handle_broadcasting(a_grad, a.shape), 
                self._handle_broadcasting(b_grad, b.shape))

class Pow(Function, Module):
    name = "Pow"
    """Element-wise power."""
    @ng.fuse
    def _fused_bw_a(grad_output, a, b):
        return grad_output * b * a ** (b - 1)
    @ng.fuse
    def _fused_bw_b(grad_output, a, b):
        return grad_output * xp.log(a) * (a ** b)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a ** b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        a_grad = Pow._fused_bw_a(grad_output, a.data, b.data) if a.requires_grad else None
        b_grad = Pow._fused_bw_b(grad_output, a.data, b.data) if b.requires_grad else None
        return (self._handle_broadcasting(a_grad, a.shape), 
                self._handle_broadcasting(b_grad, b.shape))

# Convenience functions for arithmetic operations
# These functions are designed to be used directly with Tensor objects.
def add(a, b):
    return Add()(a, b)
def sub(a, b):
    return Sub()(a, b)
def mul(a, b):
    return Mul()(a, b)
def div(a, b):
    return Div()(a, b)
def pow(a, b):
    return Pow()(a, b)