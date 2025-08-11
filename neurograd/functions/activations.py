from neurograd import xp
from neurograd.functions.base import Function
from neurograd.nn.module import Module

### Activation functions classes for Functional API
# These classes implement common activation functions used in neural networks.
class ReLU(Function, Module):
    name = "ReLU"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.maximum(0, x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * (x.data > 0) if x.requires_grad else None
        return x_grad

class Sigmoid(Function, Module):
    name = "Sigmoid"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
        self.sigmoid_x = None
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.sigmoid_x = 1 / (1 + xp.exp(-x))
        return self.sigmoid_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * self.sigmoid_x * (1 - self.sigmoid_x) if x.requires_grad else None
        return x_grad

class Softmax(Function, Module):
    name = "Softmax"
    def __init__(self, axis: int = -1, keepdims: bool = False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
        self.softmax_x = None
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        exp_x = xp.exp(x - xp.max(x, axis=self.axis, keepdims=self.keepdims))
        self.softmax_x = exp_x / xp.sum(exp_x, axis=self.axis, keepdims=self.keepdims)
        return self.softmax_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        dot_product = xp.sum(self.softmax_x * grad_output, axis=self.axis, keepdims=self.keepdims)
        x_grad = self.softmax_x * (grad_output - dot_product) if x.requires_grad else None
        return x_grad

class Tanh(Function, Module):
    name = "Tanh"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
        self.tanh_x = None
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.tanh_x = xp.tanh(x)
        return self.tanh_x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * (1 - self.tanh_x ** 2) if x.requires_grad else None
        return x_grad

class LeakyReLU(Function, Module):
    name = "LeakyReLU"
    def __init__(self, negative_slope: float = 0.01):
        Function.__init__(self)
        Module.__init__(self)
        self.negative_slope = negative_slope
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.where(x >= 0, x, self.negative_slope * x)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output * xp.where(x.data >= 0, 1, self.negative_slope) if x.requires_grad else None
        return x_grad
    

class Passthrough(Function, Module):
    name = "Passthrough"
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return x
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        x_grad = grad_output if x.requires_grad else None
        return x_grad
    

### Activation functions for user convenience
# These functions are designed to be used directly with tensors, providing a more intuitive interface.
def relu(x):
    return ReLU()(x)  
def sigmoid(x):
    return Sigmoid()(x)   
def softmax(x , axis: int = 0, keepdims: bool = True):
        return Softmax(axis = axis, keepdims = keepdims)(x)   
def tanh(x):
        return Tanh()(x)
def leaky_relu(x, negative_slope: float = 0.01):
        return LeakyReLU(negative_slope=negative_slope)(x)