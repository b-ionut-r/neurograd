from neurograd import xp
from .base import Function
from neurograd.nn.module import Module

class Sum(Function, Module):
    name = "Sum"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.sum(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Expand grad_output to match input shape
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            grad = xp.expand_dims(grad, axis=self.axis)
        
        # Broadcast to original shape
        grad = xp.broadcast_to(grad, x.data.shape)
        return grad

class Mean(Function, Module):
    name = "Mean"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.mean(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Calculate number of elements being averaged
        if self.axis is None:
            n = x.data.size
        else:
            if isinstance(self.axis, int):
                n = x.data.shape[self.axis]
            else:
                # Handle tuple of axes
                n = 1
                for ax in self.axis:
                    n *= x.data.shape[ax]
        
        # Expand and broadcast gradient
        grad = grad_output / n
        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, int):
                grad = xp.expand_dims(grad, axis=self.axis)
            else:
                # Handle multiple axes
                for ax in sorted(self.axis):
                    grad = xp.expand_dims(grad, axis=ax)
        
        grad = xp.broadcast_to(grad, x.data.shape)
        return grad

class Max(Function, Module):
    name = "Max"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.max_vals = xp.max(x, axis=self.axis, keepdims=True)
        return xp.max(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Create mask for maximum values
        mask = (x.data == self.max_vals).astype(x.data.dtype)
        
        # Handle ties by distributing gradient equally
        if self.axis is None:
            # Global max - distribute among all max elements
            mask = mask / xp.sum(mask)
        else:
            # Axis-wise max - distribute among max elements along each reduced dimension
            if isinstance(self.axis, int):
                count = xp.sum(mask, axis=self.axis, keepdims=True)
            else:
                count = xp.sum(mask, axis=self.axis, keepdims=True)
            # Avoid division by zero
            count = xp.where(count == 0, 1, count)
            mask = mask / count
        
        # Expand and broadcast gradient
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, int):
                grad = xp.expand_dims(grad, axis=self.axis)
            else:
                for ax in sorted(self.axis):
                    grad = xp.expand_dims(grad, axis=ax)
        
        grad = xp.broadcast_to(grad, x.data.shape) * mask
        return grad

class Min(Function, Module):
    name = "Min"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.min_vals = xp.min(x, axis=self.axis, keepdims=True)
        return xp.min(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Create mask for minimum values
        mask = (x.data == self.min_vals).astype(x.data.dtype)
        
        # Handle ties by distributing gradient equally
        if self.axis is None:
            # Global min - distribute among all min elements
            mask = mask / xp.sum(mask)
        else:
            # Axis-wise min - distribute among min elements along each reduced dimension
            if isinstance(self.axis, int):
                count = xp.sum(mask, axis=self.axis, keepdims=True)
            else:
                count = xp.sum(mask, axis=self.axis, keepdims=True)
            # Avoid division by zero
            count = xp.where(count == 0, 1, count)
            mask = mask / count
        
        # Expand and broadcast gradient
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, int):
                grad = xp.expand_dims(grad, axis=self.axis)
            else:
                for ax in sorted(self.axis):
                    grad = xp.expand_dims(grad, axis=ax)
        
        grad = xp.broadcast_to(grad, x.data.shape) * mask
        return grad


class Std(Function, Module):
    name = "Std"
    def __init__(self, axis=None, keepdims=False, ddof=0):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
        self.ddof = ddof  # Delta degrees of freedom
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.std(x, axis=self.axis, keepdims=self.keepdims, ddof=self.ddof)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Calculate mean and variance for gradient computation
        mean_val = xp.mean(x.data, axis=self.axis, keepdims=True)
        var_val = xp.var(x.data, axis=self.axis, keepdims=True, ddof=self.ddof)
        std_val = xp.sqrt(var_val)
        
        # Calculate number of elements
        if self.axis is None:
            n = x.data.size - self.ddof
        else:
            if isinstance(self.axis, int):
                n = x.data.shape[self.axis] - self.ddof
            else:
                n = 1
                for ax in self.axis:
                    n *= x.data.shape[ax]
                n -= self.ddof
        
        # Avoid division by zero
        std_val = xp.where(std_val == 0, 1e-8, std_val)
        
        # Gradient: d/dx std(x) = (x - mean) / (n * std)
        grad = (x.data - mean_val) / (n * std_val)
        
        # Expand grad_output to match input shape
        grad_out = grad_output
        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, int):
                grad_out = xp.expand_dims(grad_out, axis=self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_out = xp.expand_dims(grad_out, axis=ax)
        
        # Broadcast and multiply
        grad_out = xp.broadcast_to(grad_out, x.data.shape)
        grad = grad * grad_out
        
        return grad


def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)
def mean(x, axis=None, keepdims=False):
    return Mean(axis=axis, keepdims=keepdims)(x)
def max(x, axis=None, keepdims=False):
    return Max(axis=axis, keepdims=keepdims)(x)
def min(x, axis=None, keepdims=False):
    return Min(axis=axis, keepdims=keepdims)(x)
def std(x, axis=None, keepdims=False, ddof=0):
    return Std(axis=axis, keepdims=keepdims, ddof=ddof)(x)