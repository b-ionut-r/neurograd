from neurograd import xp
from typing import TYPE_CHECKING, Union, Tuple, Sequence, Literal
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from neurograd.tensor import Tensor


def conv2d(input: Union["Tensor", xp.ndarray], filters: Union["Tensor", xp.ndarray],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0):
    
    import neurograd as ng
    
    # Expand batch axis dim if needed 
    if input.ndim == 3:
        input = ng.expand_dims(input, axis=0)  # Add batch dimension
    if filters.ndim == 3:
        filters = ng.expand_dims(filters, axis=0)  # Add output channel dimension
        
    # Extract input and filters shape (channel first)
    N, C, H, W = input.shape
    F_N, F_C, F_H, F_W = filters.shape
    assert C == F_C, "Channel axis must match to convolve input with filters."    
    
    # Handle padding
    if padding == "valid":
        padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    elif padding == "same":
        pad_H = F_H - 1
        pad_W = F_W - 1
        padding = [(0, 0), (0, 0), (pad_H//2, pad_H - pad_H//2), (pad_W//2, pad_W - pad_W//2)]
    else:
        # Handle numeric padding
        if isinstance(padding, (int, float)):
            padding = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        elif len(padding) == 2:
            pad_h, pad_w = padding
            padding = [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)]  # Fixed order
    
    # Calculate output dimensions after padding
    pad_h = padding[2][0] + padding[2][1]  # Fixed indices
    pad_w = padding[3][0] + padding[3][1]
    out_H = int((H + pad_h - F_H) // strides[0]) + 1
    out_W = int((W + pad_w - F_W) // strides[1]) + 1
    
    input = ng.pad(input, pad_width=padding, mode='constant', constant_values=padding_value)
    slides = ng.sliding_window_view(input, window_shape=(F_H, F_W), strides=strides, axes=(2, 3))  # Fixed axes
    slides = ng.transpose(slides, axes=(0, 2, 3, 1, 4, 5))  # (N, out_H, out_W, C, F_H, F_W)
    slides = slides.reshape((N, out_H * out_W, C * F_H * F_W))
    filters = filters.reshape((F_N, C * F_H * F_W))
    output = ng.tensordot(slides, filters, axes=([2], [1]))  # (N, out_H * out_W, F_N)
    output = output.reshape((N, out_H, out_W, F_N)).transpose((0, 3, 1, 2))  # (N, F_N, out_H, out_W)
    return output



def pool2d(input: Union["Tensor", xp.ndarray], 
           pool_size: Union[int, Tuple[int, ...]],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0, pooling_fn = None):
    import neurograd as ng
    
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
    
    # Handle padding
    if padding == "valid":
        padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    elif padding == "same":
        pad_H = P_H - 1
        pad_W = P_W - 1
        padding = [(0, 0), (0, 0), (pad_H//2, pad_H - pad_H//2), (pad_W//2, pad_W - pad_W//2)]
    else:
        # Handle numeric padding
        if isinstance(padding, (int, float)):
            padding = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        elif len(padding) == 2:
            pad_h, pad_w = padding
            padding = [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)]
    
    # Calculate output dimensions after padding
    pad_h = padding[2][0] + padding[2][1] 
    pad_w = padding[3][0] + padding[3][1] 
    out_H = int((H + pad_h - P_H) // strides[0]) + 1
    out_W = int((W + pad_w - P_W) // strides[1]) + 1
    
    input = ng.pad(input, pad_width=padding, mode='constant', constant_values=padding_value)
    slides = ng.sliding_window_view(input, window_shape=(P_H, P_W), 
                                    strides=strides, 
                                    axes=(2, 3)) # slides shape: (N, C, out_H, out_W, P_H, P_W)
    output = pooling_fn(slides, axis=(4, 5), keepdims=False) # output shape: (N, C, out_H, out_W) # (4, 5) OR (-2, -1)
    
    return output


def maxpool2d(input: Union["Tensor", xp.ndarray], 
              pool_size: Union[int, Tuple[int, ...]],
              strides: Union[int, Tuple[int, ...]] = (1, 1),
              padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
              padding_value: Union[int, float] = 0):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.max)


def averagepool2d(input: Union["Tensor", xp.ndarray], 
              pool_size: Union[int, Tuple[int, ...]],
              strides: Union[int, Tuple[int, ...]] = (1, 1),
              padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
              padding_value: Union[int, float] = 0):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.mean)


# Set aliases
pooling2d = pool2d
maxpooling2d = maxpool2d
averagepooling2d = averagepool2d
