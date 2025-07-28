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
        
        a_grad = self._handle_broadcasting(grad_output, a.data.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(grad_output, b.data.shape) if b.requires_grad else None
        
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
        
        a_grad = self._handle_broadcasting(grad_output, a.data.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(-grad_output, b.data.shape) if b.requires_grad else None
        
        return a_grad, b_grad

class Mul(Function, Module):
    name = "Mul"
    """Element-wise multiplication."""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a * b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        
        a_grad = self._handle_broadcasting(grad_output * b.data, a.data.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(grad_output * a.data, b.data.shape) if b.requires_grad else None
        
        return a_grad, b_grad

class Div(Function, Module):
    name = "Div"
    """Element-wise division."""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a / b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        
        a_grad = self._handle_broadcasting(grad_output / b.data, a.data.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(-grad_output * a.data / (b.data ** 2), b.data.shape) if b.requires_grad else None
        
        return a_grad, b_grad

class Pow(Function, Module):
    name = "Pow"
    """Element-wise power."""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        return a ** b
    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        a, b = self.parent_tensors
        
        a_grad = self._handle_broadcasting(grad_output * b.data * a.data ** (b.data - 1), a.data.shape) if a.requires_grad else None
        b_grad = self._handle_broadcasting(grad_output * xp.log(a.data) * (a.data ** b.data), b.data.shape) if b.requires_grad else None
        
        return a_grad, b_grad
    

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