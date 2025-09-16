from neurograd import xp
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



# Implementation with cuDNN backend
class Convolver(Function):
    """2D Convolution operation using cuDNN backend for GPU acceleration."""
    name = "Convolver"
    def __init__(self, strides: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int], Literal["valid", "same"]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 depthwise: bool = False, autotune: int = 5):
        assert CUDNN_AVAILABLE, "cuDNN is required for Convolver but is not available."
        Function.__init__(self)
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding_mode = padding  # Store original for "same"/"valid" handling
        if padding == "valid":
            self.padding = (0, 0)
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding if isinstance(padding, tuple) else (0, 0)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.depthwise = depthwise
        self.autotune = autotune 
        # Initialize descriptors
        self.inputs_desc = None
        self.filters_desc = None
        self.conv_desc = None
        self.output_desc = None
        # Best algos cache
        self.algo_fw = None
        self.algo_bw_inputs = None
        self.algo_bw_filters = None
        # Create cuDNN handle
        self.handle = cudnn.create()
        
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
    
    def _create_filter_descriptor(self, array):
        """Create a cuDNN filter descriptor for the given array"""
        desc = cudnn.createFilterDescriptor()
        k, c, h, w = array.shape
        cudnn.setFilter4dDescriptor_v4(
            desc,
            self._get_cudnn_dtype(array.dtype),
            cudnn.CUDNN_TENSOR_NCHW,
            k, c, h, w
        )
        return desc

    def _create_convolution_descriptor(self, dtype):
        """Create a cuDNN convolution descriptor"""
        desc = cudnn.createConvolutionDescriptor()
        pad_h, pad_w = self.padding
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilation
        cudnn.setConvolution2dDescriptor_v5(
            desc,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            cudnn.CUDNN_CROSS_CORRELATION,
            dtype
        )
        return desc

    def _compute_output_shape(self, inputs: xp.ndarray, filters: xp.ndarray) -> Tuple[int, int, int, int]:
        """Compute the output tensor shape for convolution"""
        n, c_in, h_in, w_in = inputs.shape
        k_out, c_f, k_h, k_w = filters.shape
        pad_h, pad_w = self.padding
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilation
        h_out = ((h_in + 2 * pad_h - dilation_h * (k_h - 1) - 1) // stride_h) + 1
        w_out = ((w_in + 2 * pad_w - dilation_w * (k_w - 1) - 1) // stride_w) + 1
        if self.depthwise:
            c_out = c_in
        else:
            c_out = k_out
        return (n, c_out, h_out, w_out)
    
    def forward(self, inputs: xp.ndarray, filters: xp.ndarray) -> xp.ndarray:
        # Set filters same type as inputs to avoid type issues
        filters = filters.astype(inputs.dtype, copy=False)
        # Input validation and shape normalization
        if inputs.ndim == 3:
            inputs = xp.expand_dims(inputs, axis=0)  # (1, C, H, W)
        N, C, H, W = inputs.shape
        if not self.depthwise:
            if filters.ndim == 3:
                filters = xp.expand_dims(filters, axis=0)  # (1, C, F_H, F_W)
            F_N, F_C, F_H, F_W = filters.shape
            assert C == F_C, "Channel axis must match to convolve input with filters."
        else:
            if filters.ndim == 3:
                # Reshape (C, F_H, F_W) to (C, 1, F_H, F_W) for cuDNN depthwise
                filters = xp.expand_dims(filters, axis=1)  # (C, 1, F_H, F_W)
            F_N, F_C, F_H, F_W = filters.shape
            assert F_N == C and F_C == 1, "For depthwise convolution, filters must be (C, 1, F_H, F_W)."
        
        # Handle "same" padding
        if self.padding_mode == "same":
            sh, sw = self.strides
            out_H = (H + sh - 1) // sh
            out_W = (W + sw - 1) // sw
            pad_h = max((out_H - 1) * sh + F_H - H, 0)
            pad_w = max((out_W - 1) * sw + F_W - W, 0)
            self.padding = (pad_h // 2, pad_w // 2)
        elif self.padding_mode == "valid":
            self.padding = (0, 0)
        
        # Create tensor and filter descriptors
        self.inputs_desc = self._create_tensor_descriptor(inputs)
        self.filters_desc = self._create_filter_descriptor(filters)
        # Create convolution descriptor
        # ALWAYS: use FP32 compute type even when inputs are FP16 for numerical stability
        self.conv_desc = self._create_convolution_descriptor(cudnn.CUDNN_DATA_FLOAT)
        # Set group count for depthwise convolution
        if self.depthwise:
            cudnn.setConvolutionGroupCount(self.conv_desc, int(inputs.shape[1]))
        else:
            cudnn.setConvolutionGroupCount(self.conv_desc, 1)
        # Create output tensor
        output = xp.empty(self._compute_output_shape(inputs, filters), dtype=inputs.dtype)
        self.output_desc = self._create_tensor_descriptor(output)
        # Find the best algorithm for convolution
        if self.algo_fw is None:
            algos = cudnn.findConvolutionForwardAlgorithm(
                self.handle,
                self.inputs_desc,
                self.filters_desc,
                self.conv_desc,
                self.output_desc,            
                self.autotune,
            )
            self.algo_fw = algos[0]["algo"]
        # Get workspace size
        workspace_size = cudnn.getConvolutionForwardWorkspaceSize(
            self.handle,
            self.inputs_desc,
            self.filters_desc,
            self.conv_desc,
            self.output_desc,
            self.algo_fw
        )
        # Allocate workspace if needed
        workspace = xp.empty(workspace_size, dtype=xp.uint8) if workspace_size > 0 else None
        # Perform forward convolution
        alpha = np.array(1.0, dtype=np.float32)
        beta = np.array(0.0, dtype=np.float32)
        cudnn.convolutionForward(
            self.handle,
            alpha.ctypes.data,
            self.inputs_desc,
            inputs.data.ptr,
            self.filters_desc,
            filters.data.ptr,
            self.conv_desc,
            self.algo_fw,
            workspace.data.ptr if workspace is not None else 0,
            workspace_size,
            beta.ctypes.data,
            self.output_desc,
            output.data.ptr
        )
        return output
        

    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        inputs, filters = self.parent_tensors[0], self.parent_tensors[1]
        grad_input = self.backward_input(grad_output) if inputs.requires_grad else None
        grad_filters = self.backward_filters(grad_output) if filters.requires_grad else None
        return grad_input, grad_filters

    def backward_input(self, grad_output: xp.ndarray) -> xp.ndarray:
        inputs, filters = self.parent_tensors[0].data, self.parent_tensors[1].data
        grad_output = grad_output.astype(inputs.dtype, copy=False)
        grad_input = xp.empty_like(inputs)  # Use empty_like consistently
        
        # Find the best algorithm for backward data
        if self.algo_bw_inputs is None:
            algos = cudnn.findConvolutionBackwardDataAlgorithm(
                self.handle,
                self.filters_desc,
                self.output_desc,
                self.conv_desc,
                self.inputs_desc,
                self.autotune,
            )
            self.algo_bw_inputs = algos[0]["algo"]
        
        # Get workspace size
        workspace_size = cudnn.getConvolutionBackwardDataWorkspaceSize(
            self.handle,
            self.filters_desc,
            self.output_desc,
            self.conv_desc, 
            self.inputs_desc,
            self.algo_bw_inputs
        )
        
        # Allocate workspace
        workspace = xp.empty(workspace_size, dtype=xp.uint8) if workspace_size > 0 else None
        
        alpha = np.array(1.0, dtype=np.float32)
        beta = np.array(0.0, dtype=np.float32)
        
        # Perform backward data convolution
        cudnn.convolutionBackwardData_v3(
            self.handle,
            alpha.ctypes.data,
            self.filters_desc,
            filters.data.ptr,
            self.output_desc,
            grad_output.data.ptr,
            self.conv_desc,
            self.algo_bw_inputs,
            workspace.data.ptr if workspace is not None else 0,
            workspace_size,
            beta.ctypes.data,
            self.inputs_desc,
            grad_input.data.ptr
        )
        return grad_input

    def backward_filters(self, grad_output: xp.ndarray) -> xp.ndarray:
        inputs, filters = self.parent_tensors[0].data, self.parent_tensors[1].data
        grad_output = grad_output.astype(inputs.dtype, copy=False)
        grad_filters = xp.empty_like(filters)  # Use empty_like, not zeros_like
        
        # Find the best algorithm for backward filters
        if self.algo_bw_filters is None:
            algos = cudnn.findConvolutionBackwardFilterAlgorithm(
                self.handle,
                self.inputs_desc,
                self.output_desc,
                self.conv_desc,
                self.filters_desc,
                self.autotune,
            )
            self.algo_bw_filters = algos[0]["algo"]
        
        # Get workspace size
        workspace_size = cudnn.getConvolutionBackwardFilterWorkspaceSize(
            self.handle,
            self.inputs_desc,
            self.output_desc,
            self.conv_desc,
            self.filters_desc,
            self.algo_bw_filters
        )
        
        # Allocate workspace
        workspace = xp.empty(workspace_size, dtype=xp.uint8) if workspace_size > 0 else None
        
        alpha = np.array(1.0, dtype=np.float32)
        beta = np.array(0.0, dtype=np.float32)
        
        # Perform backward filters convolution
        cudnn.convolutionBackwardFilter_v3(
            self.handle,
            alpha.ctypes.data,
            self.inputs_desc,
            inputs.data.ptr,
            self.output_desc,
            grad_output.data.ptr,
            self.conv_desc,
            self.algo_bw_filters,
            workspace.data.ptr if workspace is not None else 0,
            workspace_size,
            beta.ctypes.data,
            self.filters_desc,
            grad_filters.data.ptr
        )
        return grad_filters




# Implementation without cuDNN (pure NeuroGrad)
def conv2d(input: Union["Tensor", xp.ndarray], filters: Union["Tensor", xp.ndarray],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0, depthwise: bool = False):
    
    import neurograd as ng
    from neurograd.functions.tensor_ops import SlidingWindowView
    if input.ndim == 3:
        input = ng.expand_dims(input, axis=0)  # (1, C, H, W)
    N, C, H, W = input.shape
    if not depthwise:
        if filters.ndim == 3:
            filters = ng.expand_dims(filters, axis=0)  # (1, C, F_H, F_W)
        F_N, F_C, F_H, F_W = filters.shape
        assert C == F_C, "Channel axis must match to convolve input with filters."
    else:
        F_N, F_H, F_W = filters.shape
        assert F_N == C, "For depthwise convolution, number of filters must match number of input channels."


    if isinstance(strides, int):
        strides = (strides, strides)
    sh, sw = strides
    if padding == "valid":
        pad_h = pad_w = 0
    elif padding == "same":
        out_H = (H + sh - 1) // sh
        out_W = (W + sw - 1) // sw
        pad_h = max((out_H - 1) * sh + F_H - H, 0)
        pad_w = max((out_W - 1) * sw + F_W - W, 0)
    elif isinstance(padding, (int, float)):
        pad_h = pad_w = int(padding) * 2
    else:
        pad_h, pad_w = padding[0] * 2, padding[1] * 2

    out_H = (H + pad_h - F_H) // sh + 1
    out_W = (W + pad_w - F_W) // sw + 1
    if pad_h > 0 or pad_w > 0:
        ph1, pw1 = pad_h // 2, pad_w // 2
        input = ng.pad(
            input,
            ((0, 0), (0, 0), (ph1, pad_h - ph1), (pw1, pad_w - pw1)),
            constant_values=padding_value,
            memsave=True
        )
        
    # Create a fresh sliding window view op per call
    slider = SlidingWindowView(window_shape=(F_H, F_W), strides=strides, axes=(2, 3))
    if not depthwise:
        slides = slider(input)  # (N, C, out_H, out_W, F_H, F_W)
        filters = filters # (F_N, C, F_H, F_W)
        output = ng.einsum("ncpqhw,fchw->nfpq", slides, filters, backend="opt_einsum") # (N, F_N, out_H, out_W) 
    else:
        slides = slider(input)  # (N, C, out_H, out_W, F_H, F_W)
        filters = filters # (C, F_H, F_W)
        output = ng.einsum('ncpqhw,chw->ncpq', slides, filters, backend="opt_einsum") # (N, C, out_H, out_W)
    return output



def pool2d(input: Union["Tensor", xp.ndarray], 
           pool_size: Union[int, Tuple[int, ...]],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0, pooling_fn = None):
    
    import neurograd as ng
    from neurograd.functions.tensor_ops import SlidingWindowView

    if pooling_fn is None:
        pooling_fn = ng.max  
    
    # Normalize params
    pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
    strides = strides if isinstance(strides, tuple) else (strides, strides)
    
    # Expand batch axis dim if needed 
    if input.ndim == 3:
        input = ng.expand_dims(input, axis=0)  # Add batch dimension
    
    # Extract input shape (NCHW format for consistency with conv2d)
    N, C, H, W = input.shape
    P_H, P_W = pool_size
    
    # Compute compact stride-aware padding
    sh, sw = strides
    if padding == "valid":
        pad_h = pad_w = 0
    elif padding == "same":
        out_H = (H + sh - 1) // sh
        out_W = (W + sw - 1) // sw
        pad_h = max((out_H - 1) * sh + P_H - H, 0)
        pad_w = max((out_W - 1) * sw + P_W - W, 0)
    elif isinstance(padding, (int, float)):
        pad_h = pad_w = int(padding) * 2
    else:
        pad_h, pad_w = padding[0] * 2, padding[1] * 2

    # Output dims (not used downstream, but documented by formula)
    out_H = (H + pad_h - P_H) // sh + 1
    out_W = (W + pad_w - P_W) // sw + 1

    # Apply symmetric padding
    if pad_h or pad_w:
        ph1, pw1 = pad_h // 2, pad_w // 2
        padding = [(0, 0), (0, 0), (ph1, pad_h - ph1), (pw1, pad_w - pw1)]
    else:
        padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    input = ng.pad(input, pad_width=padding, mode='constant', constant_values=padding_value,
                   memsave=True)

    # Create a fresh sliding window view op per call
    slider = SlidingWindowView(window_shape=(P_H, P_W), strides=strides, axes=(2, 3))
    slides = slider(input)  # (N, C, out_H, out_W, P_H, P_W)
    output = pooling_fn(slides, axis=(4, 5), keepdims=False) # output shape: (N, C, out_H, out_W) # (4, 5) OR (-2, -1)
    
    return output


def maxpool2d(input: Union["Tensor", xp.ndarray], 
              pool_size: Union[int, Tuple[int, ...]],
              strides: Union[int, Tuple[int, ...]] = (2, 2),
              padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
              padding_value: Union[int, float] = 0):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.max)


def averagepool2d(input: Union["Tensor", xp.ndarray], 
                  pool_size: Union[int, Tuple[int, ...]],
                  strides: Union[int, Tuple[int, ...]] = (2, 2),
                  padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
                  padding_value: Union[int, float] = 0):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.mean)


# Set aliases
pooling2d = pool2d
maxpooling2d = maxpool2d
averagepooling2d = averagepool2d
