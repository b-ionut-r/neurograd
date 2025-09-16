from neurograd import xp, float32
from neurograd.functions import Function
from typing import TYPE_CHECKING, Union, Tuple, Sequence, Literal
import numpy as np
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from neurograd.tensor import Tensor

if xp is not np:
    import cupy 
    try:
        from cupy.cuda import cudnn
        CUDNN_AVAILABLE = True
    except:
        CUDNN_AVAILABLE = False
else:
    CUDNN_AVAILABLE = False


class BatchNormalizerCUDNN(Function):
    def __init__(self,
                 num_features: int,
                 epsilon: float = 1e-5,
                 momentum: float = 0.9,
                 axis: Union[int, Tuple[int, ...], Sequence] = (0,),
                 memsave: bool = False,
                 dtype = float32):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis
        self.memsave = memsave
        self.dtype = dtype
        # Save running stats
        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var = xp.ones(num_features, dtype=xp.float32)
        # Handle, descriptors, etc.
        self.handle = cudnn.create()
        self.x_desc = None
        self.x_dtype = None
        self.y_desc = None
        self.bn_desc = None
        # For saving intermediate values during training
        self.save_mean = None
        self.save_inv_var = None
        
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
        if array.ndim == 4:
            N, C, H, W = array.shape
        else:
            raise ValueError(f"Expected 4D array, got {array.ndim}D")
            
        desc = cudnn.createTensorDescriptor()
        cudnn.setTensorNdDescriptor(
            desc,
            self._get_cudnn_dtype(array.dtype),
            4,  # Always 4D for cuDNN
            (N, C, H, W),
            (C*H*W, H*W, W, 1)  # Strides for NCHW
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
    

    def forward(self, x: xp.ndarray, var_scaler: xp.ndarray, 
                mean_scaler: xp.ndarray, training: bool = True) -> xp.ndarray:
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
        if training:
            self.save_mean = xp.empty(self.x_shape[1], dtype=xp.float32)
            self.save_inv_var = xp.empty(self.x_shape[1], dtype=xp.float32)
            
            # CORRECTED: Use lowercase function names and consistent parameter order
            cudnn.batchNormalizationForwardTraining(
                self.handle,
                cudnn.CUDNN_BATCHNORM_SPATIAL,
                1.0, 0.0,  # alpha, beta
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
                1.0, 0.0,
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
        
        N, C, H, W = x.shape
        
        # Create descriptors
        dy_desc = self._create_tensor_descriptor(dy)
        
        # Allocate gradients
        dx = xp.empty_like(x, dtype=self.dtype)
        dx_desc = self._create_tensor_descriptor(dx)
        
        dvar_scaler = xp.empty(C, dtype=self.dtype)
        dmean_scaler = xp.empty(C, dtype=self.dtype)
        
        # Get current scale values (var_scaler) - this is an INPUT to cuDNN
        current_var_scaler = self.parent_tensors[1].data.astype(self.dtype, copy=False)
        if current_var_scaler.ndim == 1:
            current_var_scaler = current_var_scaler.reshape(1, -1, 1, 1)
        
        cudnn.batchNormalizationBackward(
            self.handle,
            cudnn.CUDNN_BATCHNORM_SPATIAL,
            1.0, 0.0,  # alphaDataDiff, betaDataDiff
            1.0, 0.0,  # alphaParamDiff, betaParamDiff
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
        dvar_scaler = dvar_scaler.astype(self.parent_tensors[1].dtype, copy=False) if self.parent_tensors[1].requires_grad else None
        dmean_scaler = dmean_scaler.astype(self.parent_tensors[2].dtype, copy=False) if self.parent_tensors[2].requires_grad else None
        
        # Reshape gradients to match input shapes
        if dx is not None:
            dx = dx.reshape(self.original_shape)
        if dvar_scaler is not None:
            dvar_scaler = dvar_scaler.reshape(self.parent_tensors[1].shape)
        if dmean_scaler is not None:
            dmean_scaler = dmean_scaler.reshape(self.parent_tensors[2].shape)
            
        return dx, dvar_scaler, dmean_scaler
    
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