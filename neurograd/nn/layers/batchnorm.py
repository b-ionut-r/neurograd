import neurograd as ng
from neurograd import xp, CUDNN_AVAILABLE
from neurograd.nn.module import Module
# Assuming your normalize.py is available and correct
from neurograd.functions.normalize import BatchNormalizerCUDNN, BatchNormalizer
from typing import Literal, Tuple

@ng.fuse
def exp_mov_avg(running, new, momentum):
    return momentum * running + (1 - momentum) * new


class _BatchNorm(Module):
    """
    Base class for BatchNorm modules. This version is portable and robust
    by making the module always own the running statistics buffers.
    """
    def __init__(self, num_features: int, shape: Tuple, axis: Tuple, 
                 batch_momentum: float = 0.9, epsilon: float = 1e-5,
                 backend: Literal["cudnn", "xp"] = "cudnn", memsave: bool = True):
        super().__init__()
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        self.memsave = memsave

        self.add_parameter("mean_scaler", ng.zeros(shape, dtype=ng.float32, requires_grad=True)) # beta
        self.add_parameter("var_scaler",  ng.ones(shape, dtype=ng.float32, requires_grad=True))  # gamma

        # ALWAYS register running stats as buffers with a standardized 4D shape.
        cudnn_shape = (1, num_features, 1, 1)
        self.add_buffer("running_mean", ng.zeros(cudnn_shape, dtype=ng.float32, requires_grad=False))
        self.add_buffer("running_var",  ng.ones(cudnn_shape, dtype=ng.float32, requires_grad=False))

        # Instantiate the backend operator
        if backend == "cudnn" and CUDNN_AVAILABLE:
            self.backend = "cudnn"
            self._bn_op = BatchNormalizerCUDNN(
                num_features, epsilon=epsilon, momentum=batch_momentum,
                running_mean_buffer=self.running_mean.data,
                running_var_buffer=self.running_var.data
            )
        else:
            self.backend = "xp"
            self._bn_op = BatchNormalizer(axis=axis, eps=epsilon, memsave=memsave)
    
    def forward(self, X):
        from neurograd import Tensor
        is_training = self.training
        params_tuple = (self.mean_scaler, self.var_scaler)
        if self.backend == "cudnn":
            self._bn_op.set_training(is_training)
            out = self._bn_op(X, *params_tuple)
        else: # xp backend
            if is_training:
                out = self._bn_op(X, *params_tuple)
                # Reshape the batch stats to match the 4D running stats before updating
                update_shape = (1, self.num_features, 1, 1)
                batch_mean = self._bn_op.x_mean.reshape(update_shape)
                batch_var = (self._bn_op.x_std**2).reshape(update_shape)
                self.running_mean.data[:] = exp_mov_avg(self.running_mean.data, batch_mean, self.batch_momentum)
                self.running_var.data[:]  = exp_mov_avg(self.running_var.data, batch_var, self.batch_momentum)
            else: # XP INFERENCE PATH (CRITICAL FIX HERE)
                # Get the canonical shape for this specific layer's parameters
                canonical_shape = self.mean_scaler.shape
                # Reshape the 4D running stats to this canonical shape for correct broadcasting
                running_mean_reshaped = self.running_mean.data.reshape(canonical_shape)
                running_var_reshaped = self.running_var.data.reshape(canonical_shape)
                running_std = xp.sqrt(xp.maximum(running_var_reshaped, 0) + self.epsilon)
                out_data = self._bn_op._affine(X.data, running_mean_reshaped, 
                                               running_std, self.mean_scaler.data, self.var_scaler.data)
                return Tensor(out_data)
        return out

# Child classes BatchNorm and BatchNorm2D remain the same.
class BatchNorm(_BatchNorm):
    """Applies Batch Normalization for inputs shaped (N, C)."""
    def __init__(self, num_features: int, **kwargs):
        shape = (1, num_features)
        axis = (0,)
        super().__init__(num_features, shape=shape, axis=axis, **kwargs)

class BatchNorm2D(_BatchNorm):
    """Applies Batch Normalization for 4D inputs shaped (N, C, H, W)."""
    def __init__(self, num_features: int, **kwargs):
        shape = (1, num_features, 1, 1)
        axis = (0, 2, 3)
        super().__init__(num_features, shape=shape, axis=axis, **kwargs)