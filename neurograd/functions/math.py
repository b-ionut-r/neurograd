import neurograd as ng
from neurograd import xp
from .base import Function
from neurograd.nn.module import Module

# Mathematical functions classes for Functional API
class Log(Function, Module):
    name = "Log"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return grad_output / x.data if x.requires_grad else None

class Exp(Function, Module):
    name = "Exp"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output * xp.exp(x)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.exp(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Exp._fused_bw(grad_output, x.data) if x.requires_grad else None
    
class Sqrt(Function, Module):
    name = "Sqrt"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output / (2 * xp.sqrt(x))
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.sqrt(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Sqrt._fused_bw(grad_output, x.data) if x.requires_grad else None

class Cbrt(Function, Module):
    name = "Cbrt"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output / (3 * xp.cbrt(x ** 2))
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.cbrt(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Cbrt._fused_bw(grad_output, x.data) if x.requires_grad else None
    
class Sin(Function, Module):
    name = "Sin"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output * xp.cos(x)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.sin(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Sin._fused_bw(grad_output, x.data) if x.requires_grad else None

class Cos(Function, Module):
    name = "Cos"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return -grad_output * xp.sin(x)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.cos(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Cos._fused_bw(grad_output, x.data) if x.requires_grad else None

class Tan(Function, Module):
    name = "Tan"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output / (xp.cos(x) ** 2)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.tan(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Tan._fused_bw(grad_output, x.data) if x.requires_grad else None

class Log10(Function, Module):
    name = "Log10"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output / (x * xp.log(10))
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log10(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Log10._fused_bw(grad_output, x.data) if x.requires_grad else None

class Log2(Function, Module):
    name = "Log2"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output / (x * xp.log(2))
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log2(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Log2._fused_bw(grad_output, x.data) if x.requires_grad else None

class Abs(Function, Module):
    name = "Abs"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output * xp.sign(x)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.abs(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return Abs._fused_bw(grad_output, x.data) if x.requires_grad else None

class Clip(Function, Module):
    name = "Clip"
    @ng.fuse
    def _fused_bw(grad_output, x, min_v, max_v):
        mask = ((x >= min_v) & (x <= max_v))
        return grad_output * mask
    def __init__(self, min_val=None, max_val=None):
        Function.__init__(self)
        Module.__init__(self)
        self.min_val = min_val
        self.max_val = max_val
    def _bounds(self):
        min_v = self.min_val if self.min_val is not None else -xp.inf
        max_v = self.max_val if self.max_val is not None else  xp.inf
        return min_v, max_v
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        min_v, max_v = self._bounds()
        return xp.clip(x, min_v, max_v)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        min_v, max_v = self._bounds()
        return Clip._fused_bw(grad_output, x.data, min_v, max_v)

# Convenience functions for arithmetic operations
# These functions are designed to be used directly with Tensor objects.
def log(x):
    return Log()(x)
def exp(x):
    return Exp()(x)
def sin(x):
    return Sin()(x)
def cos(x):
    return Cos()(x)
def tan(x):
    return Tan()(x) 
def sqrt(x):
    return Sqrt()(x)
def cbrt(x):
    return Cbrt()(x) 
def log10(x):
    return Log10()(x) 
def log2(x):
    return Log2()(x)
def abs(x):
    return Abs()(x)
def clip(x, min_val=None, max_val=None):
    return Clip(min_val, max_val)(x)