import neurograd as ng
from neurograd import xp
from .base import Function

class BatchNormalizer(Function):
    """
    y = var_scaler * (X - mean) / sqrt(var + eps) + mean_scaler
    mean, var, mean_scaler, var_scaler are broadcastable to X and were
    computed with keepdims=True along `axes`.
    """
    name = "BatchNormalizer"

    # ---- single-output fused kernels ----
    @ng.fuse
    def _fw_fused(X, mean, inv_std, var_scaler, mean_scaler):
        return var_scaler * ((X - mean) * inv_std) + mean_scaler  # gamma * x_hat + beta

    @ng.fuse
    def _dX_fused(gY, inv_std, var_scaler):
        return gY * (var_scaler * inv_std)  # dX

    @ng.fuse
    def _dvar_term_fused(gY, x_centered, inv_std3, var_scaler):
        # gY * gamma * x_centered * (-1/2) * (var+eps)^(-3/2)
        return gY * (var_scaler * x_centered * (-0.5) * inv_std3)

    @ng.fuse
    def _dvar_scaler_term_fused(gY, x_centered, inv_std):
        # gY * x_hat
        return gY * (x_centered * inv_std)

    @ng.fuse
    def _dmean_term_fused(gY, inv_std, var_scaler):
        # gY * ( -gamma * inv_std )
        return gY * (-(var_scaler * inv_std))

    def __init__(self, axes, epsilon: float = 1e-5):
        super().__init__()
        self.axes = axes if isinstance(axes, tuple) else (axes,)
        self.epsilon = float(epsilon)

    def forward(self, X: xp.ndarray, mean: xp.ndarray, var: xp.ndarray,
                mean_scaler: xp.ndarray, var_scaler: xp.ndarray) -> xp.ndarray:
        inv_std = 1.0 / xp.sqrt(var + self.epsilon)
        # Call as class attribute to avoid implicit `self` binding
        return BatchNormalizer._fw_fused(X, mean, inv_std, var_scaler, mean_scaler)

    def backward(self, grad_output: xp.ndarray):
        X, mean, var, mean_scaler, var_scaler = self.parent_tensors
        axes = self.axes

        # Recompute intermediates (no caching from forward)
        inv_std  = 1.0 / xp.sqrt(var.data + self.epsilon)
        x_center = X.data - mean.data
        inv_std3 = inv_std ** 3  # (var+eps)^(-3/2)

        dX = (BatchNormalizer._dX_fused(grad_output, inv_std, var_scaler.data)
              if X.requires_grad else None)

        dmean_scaler = (xp.sum(grad_output, axis=axes, keepdims=True)
                        if mean_scaler.requires_grad else None)

        dvar_scaler = (xp.sum(BatchNormalizer._dvar_scaler_term_fused(grad_output, x_center, inv_std),
                              axis=axes, keepdims=True)
                       if var_scaler.requires_grad else None)

        dmean = (xp.sum(BatchNormalizer._dmean_term_fused(grad_output, inv_std, var_scaler.data),
                        axis=axes, keepdims=True)
                 if mean.requires_grad else None)

        dvar = (xp.sum(BatchNormalizer._dvar_term_fused(grad_output, x_center, inv_std3, var_scaler.data),
                       axis=axes, keepdims=True)
                if var.requires_grad else None)

        return dX, dmean, dvar, dmean_scaler, dvar_scaler


def batch_normalize(X, mean, var, mean_scaler, var_scaler, axes, epsilon=1e-5):
    return BatchNormalizer(axes, epsilon)(X, mean, var, mean_scaler, var_scaler)
