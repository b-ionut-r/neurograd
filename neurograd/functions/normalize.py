import neurograd as ng
from neurograd import xp, CUDNN_AVAILABLE
from neurograd.functions.base import Function
from typing import TYPE_CHECKING, Union, Tuple, Sequence, Literal
import numpy as np
from numpy.typing import ArrayLike
import math
import warnings
if TYPE_CHECKING:
    from neurograd.tensor import Tensor


try:
    import cupy
    from cupy.cuda import cudnn
except:
    pass



class BatchNormalizerCUDNN(Function):
    name = "BatchNormalizerCUDNN"
    def __init__(self,
                 num_features: int,
                 epsilon: float = 1e-5,
                 momentum: float = 0.9,
                 running_mean_buffer: xp.ndarray = None,
                 running_var_buffer: xp.ndarray = None,
                 dtype = "float32"):
        Function.__init__(self)
        if not CUDNN_AVAILABLE:
            raise RuntimeError("cuDNN is not available. Cannot use BatchNormalizerCUDNN.")
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.dtype = dtype
        self.training = True
        # Save running stats
        self.running_mean = running_mean_buffer if running_mean_buffer is not None \
                            else xp.zeros((1, num_features, 1, 1), dtype=xp.float32)
        self.running_var = running_var_buffer if running_var_buffer is not None \
                            else xp.ones((1, num_features, 1, 1), dtype=xp.float32)
        # Handle, descriptors, etc.
        self.handle = cudnn.create()
        self.x_desc = None
        self.x_dtype = None
        self.x_shape = None
        self.y_desc = None
        self.bn_desc = None
        # For saving intermediate values during training
        self.save_mean = None
        self.save_inv_var = None

    def set_training(self, training: bool):
        self.training = training
        
    def _get_cudnn_dtype(self, dtype):
        """Map numpy/cupy dtype to cuDNN dtype - only floating point types supported"""
        dtype_str = str(dtype)
        dtype_map = {
            'float32': cudnn.CUDNN_DATA_FLOAT,
            'float64': cudnn.CUDNN_DATA_DOUBLE,
            'float16': cudnn.CUDNN_DATA_HALF
        }
        if dtype_str in dtype_map:
            return dtype_map[dtype_str]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported types: {list(dtype_map.keys())}")

               
    def _create_tensor_descriptor(self, array):
        """Create a cuDNN tensor descriptor for the given array"""
        desc = cudnn.createTensorDescriptor()
        n, c, h, w = array.shape
        cudnn.setTensor4dDescriptor(
            desc,
            cudnn.CUDNN_TENSOR_NCHW,
            self._get_cudnn_dtype(array.dtype),
            n, c, h, w
        )
        return desc

    def _create_bn_descriptor(self):
        """Create a cuDNN normalization descriptor"""
        desc = cudnn.createTensorDescriptor()
        cudnn.deriveBNTensorDescriptor(
            desc,
            self.x_desc,
            cudnn.CUDNN_BATCHNORM_SPATIAL,
        )
        return desc
    

    def forward(self, x: xp.ndarray, mean_scaler: xp.ndarray, 
                var_scaler: xp.ndarray) -> xp.ndarray:
        """
        Forward pass for BatchNorm
        Args:
            x: Input tensor (N,C) for 1D or (N,C,H,W) for 2D
            var_scaler: Scale parameter γ (C,) or (1,C,1,1)
            mean_scaler: Bias parameter β (C,) or (1,C,1,1)
        """
        # Store original shape and reshape to 4D if needed
        self.original_shape = x.shape
        if x.ndim == 2:
            # BatchNorm1D: (N, C) → (N, C, 1, 1)
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        elif x.ndim == 4:
            # BatchNorm2D: already (N, C, H, W)
            pass
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.ndim}D")
            
        self.x_shape = x.shape  # Store 4D shape
        self.x_dtype = x.dtype
        
        # Promote to compute dtype for numerical stability
        x = x.astype(dtype=self.dtype, copy=False)
        
        # Ensure parameters are in correct shape and dtype
        if var_scaler.ndim == 1:
            var_scaler = var_scaler.reshape(1, -1, 1, 1)
        if mean_scaler.ndim == 1:
            mean_scaler = mean_scaler.reshape(1, -1, 1, 1)
            
        var_scaler = var_scaler.astype(self.dtype, copy=False)
        mean_scaler = mean_scaler.astype(self.dtype, copy=False)
        
        # Allocate output and create descriptors
        y = xp.empty_like(x, dtype=self.dtype)
        self.x_desc = self._create_tensor_descriptor(x)
        self.y_desc = self._create_tensor_descriptor(y)
        self.bn_desc = self._create_bn_descriptor()
        
        # Allocate intermediate storage for training
        alpha = np.array(1.0, dtype=np.float32)
        beta = np.array(0.0, dtype=np.float32)
        if self.training:
            self.save_mean = xp.empty((1, self.x_shape[1], 1, 1), dtype=xp.float32)
            self.save_inv_var = xp.empty((1, self.x_shape[1], 1, 1), dtype=xp.float32)
            # CORRECTED: Use lowercase function names and consistent parameter order
            cudnn.batchNormalizationForwardTraining(
                self.handle,
                cudnn.CUDNN_BATCHNORM_SPATIAL,
                alpha.ctypes.data, beta.ctypes.data,  # Pass pointers
                self.x_desc, x.data.ptr,
                self.y_desc, y.data.ptr,
                self.bn_desc,
                var_scaler.data.ptr,     # Scale (γ) comes FIRST
                mean_scaler.data.ptr,    # Bias (β) comes SECOND
                1.0 - self.momentum,     # exponentialAverageFactor
                self.running_mean.data.ptr,
                self.running_var.data.ptr,
                self.epsilon,
                self.save_mean.data.ptr,
                self.save_inv_var.data.ptr
            )
        else:
            cudnn.batchNormalizationForwardInference(
                self.handle,
                cudnn.CUDNN_BATCHNORM_SPATIAL,
                alpha.ctypes.data, beta.ctypes.data,
                self.x_desc, x.data.ptr,
                self.y_desc, y.data.ptr,
                self.bn_desc,
                var_scaler.data.ptr,     # Scale (γ) comes FIRST
                mean_scaler.data.ptr,    # Bias (β) comes SECOND
                self.running_mean.data.ptr,
                self.running_var.data.ptr,
                self.epsilon
            )
        
        # Convert back to original dtype and shape
        y = y.astype(self.x_dtype, copy=False)
        y = y.reshape(self.original_shape)
        return y
    
    def backward(self, dy: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        """Backward pass for BatchNorm"""
        # Retrieve input tensor and reshape dy to 4D if needed
        x = self.parent_tensors[0].data
        if dy.ndim == 2:
            dy = dy.reshape(dy.shape[0], dy.shape[1], 1, 1)
        
        # Promote to compute dtype
        x = x.astype(self.dtype, copy=False)
        dy = dy.astype(self.dtype, copy=False)
        
        N, C, H, W = self.x_shape
        
        # Create descriptors
        dy_desc = self._create_tensor_descriptor(dy)
        
        # Allocate gradients
        dx = xp.empty_like(x, dtype=self.dtype)
        dx_desc = self._create_tensor_descriptor(dy) # as output shaped
        
        dvar_scaler = xp.empty(C, dtype=self.dtype)
        dmean_scaler = xp.empty(C, dtype=self.dtype)
        
        # Get current scale values (var_scaler) - this is an INPUT to cuDNN
        current_var_scaler = self.parent_tensors[2].data.astype(self.dtype, copy=False)
        if current_var_scaler.ndim == 1:
            current_var_scaler = current_var_scaler.reshape(1, -1, 1, 1)
        
        alpha = np.array(1.0, dtype=np.float32)
        beta = np.array(0.0, dtype=np.float32)
        cudnn.batchNormalizationBackward(
            self.handle,
            cudnn.CUDNN_BATCHNORM_SPATIAL,
            alpha.ctypes.data, beta.ctypes.data,  # alphaDataDiff, betaDataDiff
            alpha.ctypes.data, beta.ctypes.data,  # alphaParamDiff, betaParamDiff
            self.x_desc, x.data.ptr,
            dy_desc, dy.data.ptr,
            dx_desc, dx.data.ptr,
            self.bn_desc,                    # dBnScaleBiasDesc
            current_var_scaler.data.ptr,     # bnScale (INPUT - current scale values)
            dvar_scaler.data.ptr,            # dBnScaleResult (OUTPUT - gradient w.r.t. scale)
            dmean_scaler.data.ptr,           # dBnBiasResult (OUTPUT - gradient w.r.t. bias)
            self.epsilon,
            self.save_mean.data.ptr,
            self.save_inv_var.data.ptr
        )
        
        # Convert back to original dtypes and shapes
        dx = dx.astype(self.x_dtype, copy=False) if self.parent_tensors[0].requires_grad else None
        dmean_scaler = dmean_scaler.astype(self.parent_tensors[1].dtype, copy=False) if self.parent_tensors[1].requires_grad else None
        dvar_scaler = dvar_scaler.astype(self.parent_tensors[2].dtype, copy=False) if self.parent_tensors[2].requires_grad else None
        
        # Reshape gradients to match input shapes
        if dx is not None:
            dx = dx.reshape(self.original_shape)
        if dmean_scaler is not None:
            dmean_scaler = dmean_scaler.reshape(self.parent_tensors[1].shape)
        if dvar_scaler is not None:
            dvar_scaler = dvar_scaler.reshape(self.parent_tensors[2].shape)

        return dx, dmean_scaler, dvar_scaler
    
    def __del__(self):
        """Cleanup cuDNN resources"""
        try:
            if hasattr(self, 'x_desc') and self.x_desc:
                cudnn.destroyTensorDescriptor(self.x_desc)
            if hasattr(self, 'y_desc') and self.y_desc:
                cudnn.destroyTensorDescriptor(self.y_desc)
            if hasattr(self, 'bn_desc') and self.bn_desc:
                cudnn.destroyTensorDescriptor(self.bn_desc)
            if hasattr(self, 'handle') and self.handle:
                cudnn.destroy(self.handle)
        except:
            pass  # Ignore cleanup errors



### XP ARRAY MODULE IMPLEMENTATION (NUMPY/CUPY)
# Handle CUPY BUG: Var reductions over multiple axes are slow due to a bug.
def _reduce(arr, axis, reduction_func, keepdims=False, **kwargs):
    """
    Performs a reduction iteratively over a tuple of axes for optimal speed,
    especially in CuPy. This function is the FAST PATH for decomposable
    reductions like mean, sum, max, and min.
    """
    # If axis is None, reduce over all axes by converting to a tuple of all axes.
    if axis is None:
        axis = tuple(range(arr.ndim))
    # If axis is a single integer, the library's default is already optimized.
    if isinstance(axis, int):
        return reduction_func(arr, axis=axis, keepdims=keepdims, **kwargs)
    # --- Iterative Path for Tuple of Axes (The Fast Path) ---
    ndim = arr.ndim
    axes = tuple(ax if ax >= 0 else ndim + ax for ax in axis)
    axes = tuple(sorted(axes, reverse=True))
    result = arr
    for ax in axes:
        result = reduction_func(result, axis=ax, keepdims=True, **kwargs)
    if not keepdims:
        result = xp.squeeze(result, axis=axes)
    return result

def _iterative_var(arr, axis, keepdims=False):
    """
    Calculates variance correctly and quickly using the fast `_reduce` helper.
    """
    if xp is np:
        return np.var(arr, axis=axis, keepdims=keepdims) # numpy doesn't have the bug
    arr_mean = _reduce(arr, axis, xp.mean, keepdims=True)
    arr_sq_mean = _reduce(xp.square(arr), axis, xp.mean, keepdims=True)
    var = arr_sq_mean - xp.square(arr_mean)
    var = xp.maximum(var, 0)
    if not keepdims:
        if axis is None:
            axis_to_squeeze = tuple(range(arr.ndim))
        elif isinstance(axis, int):
            axis_to_squeeze = (axis,)
        else: # tuple
            axis_to_squeeze = axis
        var = xp.squeeze(var, axis=axis_to_squeeze)
    return var



class BatchNormalizer(Function):
    name = "BatchNormalizer"
    def __init__(self, axis=0, eps=1e-5, memsave=False):
        Function.__init__(self)
        self.axis = axis
        self.eps = eps
        self.memsave = memsave
        # Cache for backward pass
        self.x_mean = None
        self.x_std = None
        self.N = None      # Cache number of elements for averaging
    def _affine(self, x: xp.ndarray, mean: xp.ndarray, std: xp.ndarray,
                mean_scaler: xp.ndarray, std_scaler: xp.ndarray):
        # Normalize
        x_hat = (x - mean) / std
        # Scale and shift
        out = std_scaler * x_hat + mean_scaler
        return out

    def forward(self, x: xp.ndarray, mean_scaler: xp.ndarray,
                std_scaler: xp.ndarray) -> xp.ndarray:
        # Calculate mean and std over the specified axes
        self.x_mean = xp.mean(x, axis=self.axis, keepdims=True)
        # Calculate variance instead of std to avoid numerical issues
        x_var = _iterative_var(x, self.axis, keepdims=True)
        self.x_std = xp.sqrt(x_var + self.eps)
        # Calculate number of elements used in normalization
        if self.N is None:
            if isinstance(self.axis, tuple):
                self.N = 1
                for ax in self.axis:
                    self.N *= x.shape[ax]
            else:
                self.N = x.shape[self.axis]
        return self._affine(x, self.x_mean, self.x_std, mean_scaler, std_scaler)
    
    def _memsave(self, parents, output):
        parents[0].data = output.data

    def backward(self, dout: xp.ndarray) -> tuple:
        x = self.parent_tensors[0].data
        mean_scaler = self.parent_tensors[1].data
        std_scaler = self.parent_tensors[2].data
        if self.memsave:
            x_hat = (x - mean_scaler) / std_scaler
        else:
            x_hat = (x - self.x_mean) / self.x_std
        # Gradients w.r.t. scale and bias parameters
        dmean_scaler = None
        dstd_scaler = None
        if self.parent_tensors[1].requires_grad:
            dmean_scaler = xp.sum(dout, axis=self.axis, keepdims=True)
        if self.parent_tensors[2].requires_grad:
            dstd_scaler = xp.sum(dout * x_hat, axis=self.axis, keepdims=True)
        # Gradient w.r.t. input x
        dx = None
        if self.parent_tensors[0].requires_grad:
            # Standard batch normalization backward pass
            dx_hat = dout * std_scaler
            # Compute gradients using the standard BatchNorm backward formula
            dx = (1.0 / self.N) * (1.0 / self.x_std) * (
                self.N * dx_hat
                - xp.sum(dx_hat, axis=self.axis, keepdims=True)
                - x_hat * xp.sum(dx_hat * x_hat, axis=self.axis, keepdims=True)
            ) 
        return dx, dmean_scaler, dstd_scaler
