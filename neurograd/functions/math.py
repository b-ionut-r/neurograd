from neurograd import xp
from .base import Function
from neurograd.nn.module import ModuleMixin

# Mathematical functions classes for Functional API
class Log(Function, ModuleMixin):
    name = "Log"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / x.data if x.requires_grad else None
        return x_grad

class Exp(Function, ModuleMixin):
    name = "Exp"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.exp(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * xp.exp(x.data) if x.requires_grad else None
        return x_grad
    
class Sqrt(Function, ModuleMixin):
    name = "Sqrt"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.sqrt(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / (2 * xp.sqrt(x.data)) if x.requires_grad else None
        return x_grad
    
class Cbrt(Function, ModuleMixin):
    name = "Cbrt"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.cbrt(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / (3 * xp.cbrt(x.data ** 2)) if x.requires_grad else None
        return x_grad    
    
class Sin(Function, ModuleMixin):
    name = "Sin"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.sin(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * xp.cos(x.data) if x.requires_grad else None
        return x_grad

class Cos(Function, ModuleMixin):
    name = "Cos"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.cos(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = -grad_output * xp.sin(x.data) if x.requires_grad else None
        return x_grad

class Tan(Function, ModuleMixin):
    name = "Tan"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.tan(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / (xp.cos(x.data) ** 2) if x.requires_grad else None
        return x_grad

class Log10(Function, ModuleMixin):
    name = "Log10"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log10(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / (x.data * xp.log(10)) if x.requires_grad else None
        return x_grad
    
class Log2(Function, ModuleMixin):
    name = "Log2"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.log2(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output / (x.data * xp.log(2)) if x.requires_grad else None
        return x_grad
    
class Abs(Function, ModuleMixin):
    name = "Abs"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.abs(x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * xp.sign(x.data) if x.requires_grad else None
        return x_grad
    

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