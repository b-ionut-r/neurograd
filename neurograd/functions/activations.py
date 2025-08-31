import neurograd as ng
from neurograd import xp
from neurograd.functions.base import Function
from neurograd.nn.module import Module


### Activation functions classes for Functional API
# These classes implement common activation functions used in neural networks.
class ReLU(Function, Module):
    name = "ReLU"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output * (x > 0)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.maximum(0, x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return ReLU._fused_bw(grad_output, x.data) if x.requires_grad else None


class ReLU6(Function, Module):
    name = "ReLU6"
    @ng.fuse
    def _fused_bw(grad_output, x):
        return grad_output * ((x > 0) & (x < 6))
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.clip(x, 0, 6)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return ReLU6._fused_bw(grad_output, x.data) if x.requires_grad else None


class Sigmoid(Function, Module):
    name = "Sigmoid"
    @ng.fuse
    def _fused_fw(x):
        return 1 / (1 + xp.exp(-x))
    @ng.fuse
    def _fused_bw(grad_output, sigmoid_x):
        return grad_output * sigmoid_x * (1 - sigmoid_x)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
        self.sigmoid_x = None
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.sigmoid_x = Sigmoid._fused_fw(x)
        return self.sigmoid_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = Sigmoid._fused_bw(grad_output, self.sigmoid_x) if x.requires_grad else None
        self.sigmoid_x = None  # Free memory
        return x_grad


class Softmax(Function, Module):
    name = "Softmax"
    def __init__(self, axis: int = -1):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.softmax_x = None  
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        x_max = xp.max(x, axis=self.axis, keepdims=True)
        exp_x = x - x_max
        xp.exp(exp_x, out=exp_x)
        exp_sum = xp.sum(exp_x, axis=self.axis, keepdims=True)
        xp.divide(exp_x, exp_sum, out=exp_x)
        self.softmax_x = exp_x
        return self.softmax_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # Compute dot product in-place to save memory
        dot_product = xp.sum(self.softmax_x * grad_output, axis=self.axis, keepdims=True)
        # Reuse grad_output memory if possible (assuming it's not needed elsewhere)
        x_grad = self.softmax_x * (grad_output - dot_product)
        return x_grad
    

class Tanh(Function, Module):
    name = "Tanh"
    @ng.fuse
    def _fused_bw(grad_output, tanh_x):
        return grad_output * (1 - tanh_x ** 2)
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
        self.tanh_x = None
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.tanh_x = xp.tanh(x)
        return self.tanh_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = Tanh._fused_bw(grad_output, self.tanh_x) if x.requires_grad else None
        self.tanh_x = None  # Free memory
        return x_grad

class LeakyReLU(Function, Module):
    name = "LeakyReLU"
    @ng.fuse
    def _fused_bw(grad_output, x, negative_slope):
        return grad_output * xp.where(x > 0, 1, negative_slope)
    def __init__(self, negative_slope: float = 0.01):
        Function.__init__(self)
        Module.__init__(self)
        self.negative_slope = negative_slope
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.where(x >= 0, x, self.negative_slope * x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return LeakyReLU._fused_bw(grad_output, x.data, self.negative_slope) if x.requires_grad else None
    

class Passthrough(Function, Module):
    name = "Passthrough"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        return grad_output if x.requires_grad else None
    

### Activation functions for user convenience
# These functions are designed to be used directly with tensors, providing a more intuitive interface.
def relu(x):
    return ReLU()(x)  
def relu6(x):
    return ReLU6()(x)
def sigmoid(x):
    return Sigmoid()(x)   
def softmax(x , axis: int = -1):
        return Softmax(axis = axis)(x)   
def tanh(x):
        return Tanh()(x)
def leaky_relu(x, negative_slope: float = 0.01):
        return LeakyReLU(negative_slope=negative_slope)(x)