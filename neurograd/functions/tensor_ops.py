from neurograd import xp
from .base import Function
from neurograd.nn.module import Module
from typing import TYPE_CHECKING, Union, Tuple, Sequence
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from neurograd.tensor import Tensor



class Reshape(Function, Module):
    name = "Reshape"
    """Reshape tensor to new shape"""
    def __init__(self, new_shape):
        Function.__init__(self)
        Module.__init__(self)
        self.new_shape = new_shape
        self.original_shape = None
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        self.original_shape = A.shape
        return xp.reshape(A, self.new_shape)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        return xp.reshape(grad_output, self.original_shape) if A.requires_grad else None


class Flatten(Function, Module):
    name = "Flatten"
    """Flatten tensor to 1D"""
    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        # Flatten all dimensions except the first (batch) dimension
        return A.reshape(A.shape[0], -1)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        return grad_output.reshape(A.shape) if A.requires_grad else None


class Squeeze(Function, Module):
    name = "Squeeze"
    """Remove dimensions of size 1 from tensor"""
    def __init__(self, axis=None):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        return xp.squeeze(A, axis=self.axis)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        return grad_output.reshape(A.shape) if A.requires_grad else None


class ExpandDims(Function, Module):
    name = "ExpandDims"
    """Add new axis of size 1 at specified position"""
    def __init__(self, axis):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
        self.original_shape = None
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        self.original_shape = A.shape
        return xp.expand_dims(A, axis=self.axis)
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        return xp.squeeze(grad_output, axis=self.axis) if A.requires_grad else None

class Concatenate(Function, Module):
    name = "Concatenate"
    def __init__(self, axis):
        Function.__init__(self)
        Module.__init__(self)
        self.axis = axis
    def forward(self, *inputs: xp.ndarray) -> xp.ndarray:
        return xp.concatenate(inputs, axis=self.axis)
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, ...]:
        inputs = self.parent_tensors
        split_indices = [tensor.shape[self.axis] for tensor in inputs]
        split_indices = xp.cumsum(split_indices)[:-1]
        split_grad = xp.split(grad_output, indices_or_sections=split_indices, axis=self.axis)
        split_grad = [g if tensor.requires_grad else None for g, tensor in zip(split_grad, inputs)]
        return tuple(split_grad)

class Cast(Function):
    """
    Cast tensor to a different dtype while maintaining autograd graph
    """
    name = "Cast"
    
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype
        self.original_dtype = None
    
    def forward(self, input_data: xp.ndarray) -> xp.ndarray:
        """Forward pass: cast data to target dtype"""
        self.original_dtype = input_data.dtype
        return input_data.astype(self.target_dtype, copy=False)
    
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray]:
        """Backward pass: cast gradient back to original dtype"""
        input_tensor = self.parent_tensors[0]
        
        if input_tensor.requires_grad:
            # Cast gradient back to original dtype for consistency
            grad_input = grad_output.astype(self.original_dtype, copy=False)
            return (grad_input,)
        else:
            return (None,)
    

class Pad(Function, Module):
    name = "Pad"
    """Pad tensor with zeros or specified value"""
    
    def __init__(self, pad_width: Union[Sequence, ArrayLike, int], mode='constant', 
                 constant_values=0, **kwargs):
        self.pad_width_input = pad_width
        self.mode = mode
        self.constant_values = constant_values
        self.kwargs = kwargs
    
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        # Normalize pad_width based on tensor dimensions
        if isinstance(self.pad_width_input, int):
            pad_width = [(self.pad_width_input, self.pad_width_input)] * A.ndim
        elif isinstance(self.pad_width_input, Sequence) and isinstance(self.pad_width_input[0], int):
            pad_width = [(p, p) for p in self.pad_width_input]
        else:
            pad_width = list(self.pad_width_input)
        
        self.pad_width = pad_width
        return xp.pad(A, pad_width=self.pad_width, mode=self.mode, 
                      constant_values=self.constant_values, **self.kwargs)
    
    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        A = self.parent_tensors[0]
        if not A.requires_grad:
            return None
        
        slices = []
        for lower, upper in self.pad_width:
            if upper == 0:
                slices.append(slice(lower, None))
            else:
                slices.append(slice(lower, -upper))
        return grad_output[tuple(slices)]



class Clone(Function, Module):
    """
    Return a copy of the input tensor that participates in autograd.
    The backward pass is identity (passes gradients through unchanged).
    """
    name = "Clone"

    def __init__(self):
        Function.__init__(self)
        Module.__init__(self)

    def forward(self, A: xp.ndarray) -> xp.ndarray:
        # Ensure data is copied so storage is independent
        return A.copy()

    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray]:
        A = self.parent_tensors[0]
        if A.requires_grad:
            return (grad_output,)
        return (None,)

class SlidingWindowView(Function, Module):
    """
    Smart Vectorized Sliding Window View with AutoDiff Support and
    sliding view buffer to avoid unnecessary memory allocation.
    """
    def __init__(self, window_shape: Sequence[int],
                 axes: Union[int, Tuple[int, ...]] = (2, 3),
                 strides: Union[int, Tuple[int, ...]] = (1, 1)):
        Function.__init__(self)
        Module.__init__(self)
        self.axes = axes if isinstance(axes, tuple) else (axes,)
        self.strides = strides if isinstance(strides, tuple) else \
                       tuple(strides for _ in range(len(self.axes)))
        self.window_shape = window_shape if isinstance(window_shape, tuple) else \
                           tuple(window_shape for _ in range(len(axes)))
        self._grad_buffer = None
        self._grad_view = None
        
    def forward(self, A: xp.ndarray) -> xp.ndarray:
        self.input_shape = A.shape
        # Build slices
        slices = [slice(None)] * A.ndim
        for ax, stride in zip(self.axes, self.strides):
            slices[ax] = slice(None, None, stride)
        self.slices = tuple(slices)
        return xp.lib.stride_tricks.sliding_window_view(
            A, self.window_shape, self.axes)[self.slices]
    
    
    def backward(self, grad_output):
        # Reuse gradient buffer if shape matches, otherwise create new
        if self._grad_buffer is None or self._grad_buffer.shape != self.input_shape:
            self._grad_buffer = xp.zeros(self.input_shape, dtype=grad_output.dtype)
            # Cache the view as well
            self._grad_view = xp.lib.stride_tricks.sliding_window_view(
                self._grad_buffer, self.window_shape, self.axes)[self.slices]
        else:
            # Just zero out the existing buffer - much faster!
            self._grad_buffer.fill(0)
        # Accumulate gradients using cached view
        self._grad_view += grad_output
        return self._grad_buffer



def reshape(A, new_shape):
    return Reshape(new_shape)(A)
def flatten(A):
    return Flatten()(A)
def squeeze(A, axis=None):
    return Squeeze(axis)(A)
def expand_dims(A, axis):
    return ExpandDims(axis)(A)
def concat(tensors: Sequence["Tensor"], axis: int) -> "Tensor":
    return Concatenate(axis=axis)(*tensors)
def cast(A, target_dtype):
    return Cast(target_dtype)(A)
def pad(A, pad_width, mode='constant', constant_values=0, **kwargs):
    return Pad(pad_width, mode, constant_values, **kwargs)(A)
def sliding_window_view(A, window_shape: Sequence[int], axes: Union[int, Tuple[int, ...]] = (2, 3), 
                        strides: Union[int, Tuple[int, ...]] = (1, 1)):
    return SlidingWindowView(window_shape, axes, strides)(A)
def clone(A):
    return Clone()(A)

# newaxis constant for numpy-style indexing
newaxis = None
    
