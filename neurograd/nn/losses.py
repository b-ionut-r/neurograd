from neurograd import xp
from neurograd.functions.base import Function
from neurograd.functions.activations import Sigmoid, Softmax
from .module import ModuleMixin
from typing import Tuple, Optional


### Loss functions with Functional Class API

class MSE(Function, ModuleMixin):
    name = "MSE"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    
    def forward(self, y_true: xp.ndarray, y_pred: xp.ndarray) -> xp.ndarray:
        return xp.mean((y_true - y_pred) ** 2)
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        y_true, y_pred = self.parent_tensors
        n_elements = y_true.data.size
        grad_y_true = 2 * grad_output * (y_true.data - y_pred.data) / n_elements if y_true.requires_grad else None
        grad_y_pred = 2 * grad_output * (y_pred.data - y_true.data) / n_elements if y_pred.requires_grad else None
        return grad_y_true, grad_y_pred
    

class RMSE(Function, ModuleMixin):
    name = "RMSE"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
        self.rmse = None
    
    def forward(self, y_true: xp.ndarray, y_pred: xp.ndarray) -> xp.ndarray:
        self.rmse = xp.sqrt(xp.mean((y_true - y_pred) ** 2))
        return self.rmse
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        y_true, y_pred = self.parent_tensors
        n_elements = y_true.data.size
        # Avoid division by zero
        rmse_safe = xp.maximum(self.rmse, 1e-8)
        grad_y_true = grad_output * (y_true.data - y_pred.data) / (rmse_safe * n_elements) if y_true.requires_grad else None
        grad_y_pred = grad_output * (y_pred.data - y_true.data) / (rmse_safe * n_elements) if y_pred.requires_grad else None
        return grad_y_true, grad_y_pred


class MAE(Function, ModuleMixin):
    name = "MAE"
    def __init__(self):
        Function.__init__(self)
        ModuleMixin.__init__(self)
    
    def forward(self, y_true: xp.ndarray, y_pred: xp.ndarray) -> xp.ndarray:
        return xp.mean(xp.abs(y_true - y_pred))
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        y_true, y_pred = self.parent_tensors
        n_elements = y_true.data.size  # Fixed typo
        diff = y_true.data - y_pred.data
        sgn = xp.sign(diff)
        grad_y_true = grad_output * sgn / n_elements if y_true.requires_grad else None
        grad_y_pred = -grad_output * sgn / n_elements if y_pred.requires_grad else None
        return grad_y_true, grad_y_pred
    

class BinaryCrossEntropy(Function, ModuleMixin):
    name = "BinaryCrossEntropy"
    def __init__(self, from_logits=False, epsilon=1e-7):
        Function.__init__(self)
        ModuleMixin.__init__(self)
        self.from_logits = from_logits
        self.epsilon = epsilon
    
    def forward(self, y_true: xp.ndarray, y_pred: xp.ndarray) -> xp.ndarray:
        if self.from_logits:
            sigmoid_op = Sigmoid()
            y_pred = sigmoid_op(y_pred).data
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = xp.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        return -xp.mean(
            y_true * xp.log(y_pred_clipped) + 
            (1 - y_true) * xp.log(1 - y_pred_clipped)
        )
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        y_true, y_pred = self.parent_tensors
        batch_size = y_true.data.size
        
        if self.from_logits:
            # For logits, gradient simplifies to (sigmoid(y_pred) - y_true) / batch_size
            sigmoid_op = Sigmoid()
            y_pred_sigmoid = sigmoid_op(y_pred).data
            y_true_grad = None  # Usually y_true doesn't require gradients
            y_pred_grad = grad_output * (y_pred_sigmoid - y_true.data) / batch_size if y_pred.requires_grad else None
        else:
            # Clip predictions to prevent division by zero
            y_pred_clipped = xp.clip(y_pred.data, self.epsilon, 1 - self.epsilon)
            y_true_grad = -grad_output * (xp.log(y_pred_clipped) - xp.log(1 - y_pred_clipped)) / batch_size if y_true.requires_grad else None
            y_pred_grad = -grad_output * (y_true.data / y_pred_clipped - (1 - y_true.data) / (1 - y_pred_clipped)) / batch_size if y_pred.requires_grad else None
        
        return y_true_grad, y_pred_grad


class CategoricalCrossEntropy(Function, ModuleMixin):
    name = "CategoricalCrossEntropy"
    def __init__(self, from_logits=False, epsilon=1e-7):
        Function.__init__(self)
        ModuleMixin.__init__(self)
        self.from_logits = from_logits
        self.epsilon = epsilon
    
    def forward(self, y_true: xp.ndarray, y_pred: xp.ndarray) -> xp.ndarray:
        if self.from_logits:
            softmax_op = Softmax()
            y_pred = softmax_op(y_pred).data
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = xp.clip(y_pred, self.epsilon, 1.0)
        
        return -xp.mean(xp.sum(y_true * xp.log(y_pred_clipped), axis=1))
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        y_true, y_pred = self.parent_tensors
        batch_size = y_true.data.shape[0]
        
        if self.from_logits:
            # For softmax + cross entropy, gradient simplifies to (y_pred - y_true) / batch_size
            softmax_op = Softmax()
            y_pred_softmax = softmax_op(y_pred).data
            y_true_grad = None  # Usually y_true doesn't require gradients
            y_pred_grad = grad_output * (y_pred_softmax - y_true.data) / batch_size if y_pred.requires_grad else None
        else:
            # Clip predictions to prevent division by zero
            y_pred_clipped = xp.clip(y_pred.data, self.epsilon, 1.0)
            y_true_grad = -grad_output * xp.log(y_pred_clipped) / batch_size if y_true.requires_grad else None
            y_pred_grad = -grad_output * y_true.data / (y_pred_clipped * batch_size) if y_pred.requires_grad else None
        
        return y_true_grad, y_pred_grad


# Functional API
def mse(y_true, y_pred):
    """Mean Squared Error Loss"""
    return MSE()(y_true, y_pred)
def rmse(y_true, y_pred):
    """Root Mean Squared Error Loss"""
    return RMSE()(y_true, y_pred)
def mae(y_true, y_pred):
    """Mean Absolute Error Loss"""
    return MAE()(y_true, y_pred)
def binary_crossentropy(y_true, y_pred, from_logits=False):
    """Binary Cross Entropy Loss"""
    return BinaryCrossEntropy(from_logits=from_logits)(y_true, y_pred)
def categorical_crossentropy(y_true, y_pred, from_logits=False):
    """Categorical Cross Entropy Loss"""
    return CategoricalCrossEntropy(from_logits=from_logits)(y_true, y_pred)