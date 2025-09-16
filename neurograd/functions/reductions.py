from neurograd import xp
import builtins
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
        return xp.sum(x, axis=self.axis, keepdims=self.keepdims, dtype=xp.float32)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # For Sum, we just need to broadcast grad_output to input shape
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            # Add dimensions back by expanding to match what keepdims=True would give
            for ax in sorted(self.axis if isinstance(self.axis, tuple) else (self.axis,)):
                # Normalize negative axes
                ax_norm = ax if ax >= 0 else len(x.shape) + ax
                grad = xp.expand_dims(grad, axis=ax_norm)
        # Broadcast to original shape
        grad = xp.broadcast_to(grad, x.shape)
        return grad

class Mean(Function, Module):
    name = "Mean"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return xp.mean(x, axis=self.axis, keepdims=self.keepdims, dtype=xp.float32)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # Calculate number of elements being averaged
        if self.axis is None:
            n = x.size
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            n = 1
            for ax in axes:
                ax_norm = ax if ax >= 0 else len(x.shape) + ax
                n *= x.shape[ax_norm]
        
        # Expand grad_output if keepdims=False
        go = grad_output
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in sorted(axes):
                ax_norm = ax if ax >= 0 else len(x.shape) + ax
                go = xp.expand_dims(go, axis=ax_norm)
        
        # Broadcast and scale by 1/n - avoid creating intermediate copy
        go = xp.broadcast_to(go, x.shape)
        return go / n

class Max(Function, Module):
    name = "Max"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Cache with keepdims=True to avoid recomputation in backward
        self.max_vals = xp.max(x, axis=self.axis, keepdims=True)
        if self.keepdims:
            return self.max_vals
        else:
            # Convert axis to positive indices for squeezing
            if self.axis is None:
                return self.max_vals.reshape(())
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)
            return xp.squeeze(self.max_vals, axis=axes)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # Create gradient mask for max elements, splitting ties evenly along the axis.
        # Use self.max_vals which already has keepdims=True for broadcasting
        mask = (x.data == self.max_vals).astype(x.data.dtype)
        if self.axis is None:
            mask /= xp.sum(mask)  # In-place division
        else:
            count = xp.sum(mask, axis=self.axis, keepdims=True)
            count = xp.where(count == 0, 1, count)
            mask /= count  # In-place division
        # Expand and broadcast gradient
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in sorted(axes):
                ax_norm = ax if ax >= 0 else len(x.shape) + ax
                grad = xp.expand_dims(grad, axis=ax_norm)
        grad = xp.broadcast_to(grad, x.shape)
        grad = grad * mask  # Use explicit multiplication to avoid in-place modification
        return grad

class Min(Function, Module):
    name = "Min"
    def __init__(self, axis=None, keepdims=False):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Cache with keepdims=True to avoid recomputation in backward
        self.min_vals = xp.min(x, axis=self.axis, keepdims=True)
        if self.keepdims:
            return self.min_vals
        else:
            # Convert axis to positive indices for squeezing
            if self.axis is None:
                return self.min_vals.reshape(())
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)
            return xp.squeeze(self.min_vals, axis=axes)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # Create gradient mask for min elements, splitting ties evenly along the axis.
        # Use self.min_vals which already has keepdims=True for broadcasting  
        mask = (x.data == self.min_vals).astype(x.data.dtype)
        if self.axis is None:
            mask /= xp.sum(mask)  # In-place division
        else:
            count = xp.sum(mask, axis=self.axis, keepdims=True)
            count = xp.where(count == 0, 1, count)
            mask /= count  # In-place division
        # Expand and broadcast gradient
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in sorted(axes):
                ax_norm = ax if ax >= 0 else len(x.shape) + ax
                grad = xp.expand_dims(grad, axis=ax_norm)
        grad = xp.broadcast_to(grad, x.shape)
        grad = grad * mask  # Use explicit multiplication to avoid in-place modification
        return grad

class Std(Function, Module):
    name = "Std"
    def __init__(self, axis=None, keepdims=False, eps=1e-8):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
        self.eps = eps
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Use stable two-pass algorithm for variance, then sqrt
        self.mean = xp.mean(x, axis=self.axis, keepdims=True, dtype=xp.float32)
        centered = x - self.mean
        self.var = xp.mean(centered * centered, axis=self.axis, keepdims=True, dtype=xp.float32)
        # Add eps to prevent sqrt of negative values due to numerical errors
        self.std_vals = xp.sqrt(xp.maximum(self.var, 0.0) + self.eps)
        if self.keepdims:
            return self.std_vals
        else:
            # Convert axis to positive indices for squeezing
            if self.axis is None:
                return self.std_vals.reshape(())
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)
            return xp.squeeze(self.std_vals, axis=axes)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        # Use cached values to avoid recomputation
        mean = self.mean
        std = self.std_vals
        # Count elements along the reduced axes (population)
        if self.axis is None:
            n = x.size
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            n = 1
            for ax in axes:
                ax_norm = ax if ax >= 0 else x.ndim + ax
                n *= x.shape[ax_norm]
        # Avoid divide-by-zero
        std_safe = xp.where(std == 0, self.eps, std)
        base_grad = (x.data - mean) / (n * std_safe)
        # If keepdims=False, expand grad_output back along reduced axes
        go = grad_output
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in sorted(axes):
                ax_norm = ax if ax >= 0 else x.ndim + ax
                go = xp.expand_dims(go, axis=ax_norm)
        # Broadcast and apply chain rule
        go = xp.broadcast_to(go, x.shape)
        base_grad = base_grad * go  # Use explicit multiplication to avoid in-place modification
        return base_grad

class Var(Function, Module):
    name = "Var"
    def __init__(self, axis=None, keepdims=False, ddof=0, eps=1e-8):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.keepdims = keepdims
        self.ddof = ddof
        self.eps = eps
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Use stable two-pass algorithm: var = E[(X - mean)²] instead of E[X²] - E[X]²
        mean_vals = xp.mean(x, axis=self.axis, keepdims=True, dtype=xp.float32)
        centered = x - mean_vals
        var_vals = xp.mean(centered * centered, axis=self.axis, keepdims=True, dtype=xp.float32)
        
        # Apply Bessel's correction if needed
        if self.ddof > 0:
            if self.axis is None:
                n = x.size
            else:
                axes = (self.axis,) if isinstance(self.axis, int) else self.axis
                n = 1
                for ax in axes:
                    ax_norm = ax if ax >= 0 else x.ndim + ax
                    n *= x.shape[ax_norm]
            var_vals = var_vals * n / (n - self.ddof)
        
        # Cache for backward pass
        self.mean = mean_vals
        
        # Handle keepdims
        if not self.keepdims and self.axis is not None:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)
            var_vals = xp.squeeze(var_vals, axis=axes)
        elif not self.keepdims and self.axis is None:
            var_vals = var_vals.reshape(())
            
        return var_vals
        
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x = self.parent_tensors[0]
        if not x.requires_grad:
            return None
        
        # Count elements along the reduced axes
        if self.axis is None:
            n = x.size
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            n = 1
            for ax in axes:
                ax_norm = ax if ax >= 0 else x.ndim + ax
                n *= x.shape[ax_norm]
        
        # Denominator for Bessel correction
        denom = builtins.max(n - self.ddof, self.eps)
        scale_factor = 2.0 / denom
        
        # Expand grad_output if keepdims=False
        go = grad_output
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in sorted(axes):
                ax_norm = ax if ax >= 0 else x.ndim + ax
                go = xp.expand_dims(go, axis=ax_norm)
        
        # Broadcast grad_output to input shape and compute gradient efficiently
        go = xp.broadcast_to(go, x.shape)
        
        # Compute gradient: d/dx var(x) = 2(x - mean)/(n - ddof)
        # Optimize to avoid creating large temporary arrays
        centered_scaled = (x.data - self.mean) * scale_factor
        grad = go * centered_scaled
        
        return grad


def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)
def mean(x, axis=None, keepdims=False):
    return Mean(axis=axis, keepdims=keepdims)(x)
def max(x, axis=None, keepdims=False):
    return Max(axis=axis, keepdims=keepdims)(x)
def min(x, axis=None, keepdims=False):
    return Min(axis=axis, keepdims=keepdims)(x)
def std(x, axis=None, keepdims=False):
    return Std(axis=axis, keepdims=keepdims)(x)
def var(x, axis=None, keepdims=False):
    return Var(axis=axis, keepdims=keepdims)(x)