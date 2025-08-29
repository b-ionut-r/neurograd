from neurograd import xp
from typing import TYPE_CHECKING, Union, Tuple, Sequence, Literal
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from neurograd.tensor import Tensor



def conv2d(input: Union["Tensor", xp.ndarray], filters: Union["Tensor", xp.ndarray],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0, depthwise: bool = False, slider = None):
    
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
        )
        
    if slider is None:
        slider = SlidingWindowView(window_shape=(F_H, F_W), strides=strides, axes=(2, 3))
    else:
        slider.window_shape = (F_H, F_W)
        slider.strides = strides
        slider.axes = (2, 3)
    if not depthwise:
        slides = slider(input)  # (N, C, out_H, out_W, F_H, F_W)
        filters = filters # (F_N, C, F_H, F_W)
        output = ng.tensordot(slides, filters, axes=[[1, 4, 5], [1, 2, 3]])  # (N, out_H, out_W, F_N)
        output = ng.transpose(output, axes=(0, 3, 1, 2))  # (N, F_N, out_H, out_W)
    else:
        slides = slider(input)  # (N, C, out_H, out_W, F_H, F_W)
        slides = ng.transpose(slides, axes=(0, 2, 3, 1, 4, 5))  # (N, out_H, out_W, C, F_H, F_W)
        filters = ng.expand_dims(ng.expand_dims(ng.expand_dims(filters, axis=0), axis=0), axis=0)  # (1, 1, 1, C, F_H, F_W)
        output = ng.sum(slides * filters, axis=(4, 5))  # (N, out_H, out_W, C)
        output = ng.transpose(output, axes=(0, 3, 1, 2))  # (N, C, out_H, out_W)

    return output



def pool2d(input: Union["Tensor", xp.ndarray], 
           pool_size: Union[int, Tuple[int, ...]],
           strides: Union[int, Tuple[int, ...]] = (1, 1),
           padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
           padding_value: Union[int, float] = 0, pooling_fn = None, slider=None):
    
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
    input = ng.pad(input, pad_width=padding, mode='constant', constant_values=padding_value)

    if slider is None:
        slider = SlidingWindowView(window_shape=(P_H, P_W), strides=strides, axes=(2, 3))
    else:
        slider.window_shape = (P_H, P_W)
        slider.strides = strides
        slider.axes = (2, 3)
    slides = slider(input)  # (N, C, out_H, out_W, P_H, P_W)
    output = pooling_fn(slides, axis=(4, 5), keepdims=False) # output shape: (N, C, out_H, out_W) # (4, 5) OR (-2, -1)
    
    return output


def maxpool2d(input: Union["Tensor", xp.ndarray], 
              pool_size: Union[int, Tuple[int, ...]],
              strides: Union[int, Tuple[int, ...]] = (2, 2),
              padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
              padding_value: Union[int, float] = 0, slider=None):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.max, slider)


def averagepool2d(input: Union["Tensor", xp.ndarray], 
                  pool_size: Union[int, Tuple[int, ...]],
                  strides: Union[int, Tuple[int, ...]] = (2, 2),
                  padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
                  padding_value: Union[int, float] = 0, slider=None):
    import neurograd as ng
    return pool2d(input, pool_size, strides, padding, padding_value, ng.mean, slider)


# Set aliases
pooling2d = pool2d
maxpooling2d = maxpool2d
averagepooling2d = averagepool2d


