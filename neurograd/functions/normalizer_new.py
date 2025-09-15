import neurograd as ng
from neurograd import xp
from .base import Function

class BatchNormalizer(Function):
    name = "BatchNormalizer"

    def __init__(self, axes, epsilon: float = 1e-5):
        super().__init__()
        self.axes = axes if isinstance(axes, tuple) else (axes,)
        self.epsilon = float(epsilon)

    def forward(self, X: xp.ndarray, mean: xp.ndarray, var: xp.ndarray,
                mean_scaler: xp.ndarray, var_scaler: xp.ndarray) -> xp.ndarray:
        # Compute in higher precision (float32) for numerical stability
        inv_std = xp.asarray(1.0 / xp.sqrt(var + self.epsilon), dtype=xp.float32)
        x_center = xp.asarray(X - mean, dtype=xp.float32)
        var_scaler_f32 = xp.asarray(var_scaler, dtype=xp.float32)
        mean_scaler_f32 = xp.asarray(mean_scaler, dtype=xp.float32)
        
        # Store for backward pass
        self.inv_std = inv_std
        self.x_center = x_center
        self.var_scaler_f32 = var_scaler_f32
        
        # Compute output
        output = var_scaler_f32 * (x_center * inv_std) + mean_scaler_f32
        return output.astype(X.dtype, copy=False)

    def backward(self, grad_output: xp.ndarray):
        X, mean, var, mean_scaler, var_scaler = self.parent_tensors
        axes = self.axes
        
        # Cast grad_output to float32 for consistent computation
        gY = xp.asarray(grad_output, dtype=xp.float32)
        
        # Precompute frequently used terms
        var_scaler_inv_std = self.var_scaler_f32 * self.inv_std
        var_scaler_inv_std3 = self.var_scaler_f32 * (self.inv_std ** 3) * (-0.5)
        
        # Compute gradients
        dX = (gY * var_scaler_inv_std).astype(X.dtype) if X.requires_grad else None
        
        dmean_scaler = xp.sum(gY, axis=axes, keepdims=True) if mean_scaler.requires_grad else None
        
        dvar_scaler = (xp.sum(gY * self.x_center * self.inv_std, axis=axes, keepdims=True)
                       if var_scaler.requires_grad else None)
        
        dmean = (-xp.sum(gY * var_scaler_inv_std, axis=axes, keepdims=True)
                 if mean.requires_grad else None)
        
        dvar = (xp.sum(gY * self.x_center * var_scaler_inv_std3, axis=axes, keepdims=True)
                if var.requires_grad else None)
        
        return dX, dmean, dvar, dmean_scaler, dvar_scaler

def batch_normalize(X, mean, var, mean_scaler, var_scaler, axes, epsilon=1e-5):
    return BatchNormalizer(axes, epsilon)(X, mean, var, mean_scaler, var_scaler)