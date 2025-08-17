from . import xp
import numpy as real_np
from typing import Optional, List, Callable, Union, Tuple, Sequence
from numpy.typing import ArrayLike


class Tensor:
    id = 0 # Unique identifier for each tensor

    def __init__(self, data, requires_grad: bool = False,
                 grad_fn: Optional[Callable] = None, name: Optional[str] = None,
                 dtype: Optional[str] = None):
        if not isinstance(data, xp.ndarray):
            self.data = xp.array(data, dtype=dtype)
        elif dtype is not None:
            self.data = data.astype(dtype, copy=False)
        else:
            self.data = data
        self.requires_grad = requires_grad # whether to compute gradients for this tensor
        self.grad = None # gradient of this tensor
        self.grad_fn = grad_fn # function that created this tensor
        if name:
            self.name = name
        else:
            self.name = f"UnnamedTensor_{Tensor.id}"
            Tensor.id += 1
        self.device = 'cpu' if xp is real_np else 'cuda'
        self.dtype = dtype or self.data.dtype


    def backward(self, grad = None, retain_graph: bool = False):
        """
        Compute gradients using automatic differentiation.
        
        Args:
            grad: Initial gradient. If None, assumes scalar output (gradient of 1).
            retain_graph: If True, the graph is retained for multiple backward passes.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot do backprop for a tensor that does not require grad.")
        
        # Check if this is a scalar (for default case)
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() can only be called for scalar outputs. "
                    "For non-scalar outputs, gradient must be provided."
                )
            grad = xp.ones_like(self.data)
        
        # Build the computational graph using topological sort
        topo_order = []
        visited = set()
        
        def build_topo(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            
            # Only process if this tensor has a grad_fn (not a leaf)
            if tensor.grad_fn is not None:
                # Visit all parent tensors first
                for parent in tensor.grad_fn.parent_tensors:
                    if parent.requires_grad:
                        build_topo(parent)
                
                topo_order.append(tensor)
        
        # Build topological ordering starting from self
        build_topo(self)
        
        # Initialize gradient for the output tensor
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad
        
        # Backpropagate in reverse topological order
        for tensor in reversed(topo_order):
            # Skip if no gradient function (shouldn't happen due to build_topo logic)
            if tensor.grad_fn is None:
                continue
                
            # Get gradient w.r.t. this tensor
            grad_output = tensor.grad
            
            # Compute gradients w.r.t. parent tensors
            parent_grads = tensor.grad_fn.backward(grad_output)
            
            # Handle single gradient return (convert to tuple)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
            
            # Accumulate gradients for parent tensors
            for parent_tensor, parent_grad in zip(tensor.grad_fn.parent_tensors, parent_grads):
                if parent_tensor.requires_grad and parent_grad is not None:
                    if parent_tensor.grad is None:
                        parent_tensor.grad = parent_grad
                    else:
                        parent_tensor.grad = parent_tensor.grad + parent_grad

            
            # Clear intermediate results to save memory (unless retaining graph)
            if not retain_graph:
                # Clear gradient for intermediate tensors (non-leaf nodes) 
                # Leaf tensors (weights/biases) keep their gradients for optimizer
                is_leaf = tensor.grad_fn is None  # Check before clearing grad_fn
                tensor.grad_fn = None
                if not is_leaf:  # Only clear grads for intermediate tensors
                    tensor.grad = None


    def cast(self, dtype):
        try: 
            # Check if already the correct dtype
            if self.data.dtype == dtype:
                return self
            
            # Use proper autograd casting operation
            from neurograd.functions.tensor_ops import Cast
            cast_op = Cast(target_dtype=dtype)
            return cast_op(self)
        except Exception as e:
            raise TypeError(f"{dtype} isn't a supported data type for the array module: {e}.")
    
    def to_half(self):
        """Convenience method to cast to FP16"""
        return self.cast(xp.float16)
    
    def to_float(self):
        """Convenience method to cast to FP32"""
        return self.cast(xp.float32)
    
    def is_half(self) -> bool:
        """Check if tensor is in FP16"""
        return self.data.dtype == xp.float16
    
    def is_float(self) -> bool:
        """Check if tensor is in FP32"""
        return self.data.dtype == xp.float32
    
    def to_double(self):
        """Convenience method to cast to FP64"""
        return self.cast(xp.float64)
    
    def is_double(self) -> bool:
        """Check if tensor is in FP64"""
        return self.data.dtype == xp.float64
    
    def to_dtype(self, dtype):
        """Convenience method to cast to specified dtype"""
        if isinstance(dtype, str):
            dtype = getattr(xp, dtype)
        return self.cast(dtype)
    
    def is_floating_point(self) -> bool:
        """Check if tensor has floating point dtype"""
        return self.data.dtype.kind == 'f'
    
    def is_integer(self) -> bool:
        """Check if tensor has integer dtype"""
        return self.data.dtype.kind in ['i', 'u']
    
    def get_dtype(self):
        """Get the dtype of the tensor"""
        return self.data.dtype
         
    
    def zero_grad(self):
        self.grad = None  # Reset gradient to None
    
    def __add__(self, other) -> 'Tensor':
        from .functions.arithmetic import Add
        return Add()(self, other)
    
    def __radd__(self, other) -> 'Tensor':
        # For right addition (e.g., 5 + tensor)
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__add__(self)
    
    def __sub__(self, other) -> 'Tensor':
        from .functions.arithmetic import Sub
        return Sub()(self, other)
    
    def __rsub__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__sub__(self)
    
    def __mul__(self, other) -> 'Tensor':
        from .functions.arithmetic import Mul
        return Mul()(self, other)
    
    def __rmul__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__mul__(self)
    
    def __truediv__(self, other) -> 'Tensor':
        from .functions.arithmetic import Div
        return Div()(self, other)
    
    def __rtruediv__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__truediv__(self)
    
    def __div__(self, other) -> 'Tensor':
        return self.__truediv__(other)
    
    def __pow__(self, other) -> 'Tensor':
        from .functions.arithmetic import Pow
        return Pow()(self, other)

    def __rpow__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__pow__(self)
    
    def __neg__(self) -> 'Tensor':
        return self * Tensor(xp.array(-1.0), requires_grad=self.requires_grad)
    
    def __matmul__(self, other) -> 'Tensor':
        from .functions.linalg import MatMul
        return MatMul()(self, other)
    
    def __rmatmul__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(xp.array(other), requires_grad=False)
        return other.__matmul__(self)
    
    def __eq__(self, other) -> 'Tensor':
        """Element-wise equality comparison."""
        if isinstance(other, Tensor):
            result = self.data == other.data
        else:
            result = self.data == other
        return Tensor(result, requires_grad=False, name=f"{self.name}_eq")
    
    def __ne__(self, other) -> 'Tensor':
        """Element-wise not-equal comparison."""
        if isinstance(other, Tensor):
            result = self.data != other.data
        else:
            result = self.data != other
        return Tensor(result, requires_grad=False, name=f"{self.name}_ne")
    
    def __lt__(self, other) -> 'Tensor':
        """Element-wise less-than comparison."""
        if isinstance(other, Tensor):
            result = self.data < other.data
        else:
            result = self.data < other
        return Tensor(result, requires_grad=False, name=f"{self.name}_lt")
    
    def __le__(self, other) -> 'Tensor':
        """Element-wise less-than-or-equal comparison."""
        if isinstance(other, Tensor):
            result = self.data <= other.data
        else:
            result = self.data <= other
        return Tensor(result, requires_grad=False, name=f"{self.name}_le")
    
    def __gt__(self, other) -> 'Tensor':
        """Element-wise greater-than comparison."""
        if isinstance(other, Tensor):
            result = self.data > other.data
        else:
            result = self.data > other
        return Tensor(result, requires_grad=False, name=f"{self.name}_gt")
    
    def __ge__(self, other) -> 'Tensor':
        """Element-wise greater-than-or-equal comparison."""
        if isinstance(other, Tensor):
            result = self.data >= other.data
        else:
            result = self.data >= other
        return Tensor(result, requires_grad=False, name=f"{self.name}_ge")
    
    def dot(self, other) -> 'Tensor':
        return self.__matmul__(other)
    
    def tensordot(self, other, axes) -> 'Tensor':
        from .functions.linalg import TensorDot
        return TensorDot(axes=axes)(self, other)

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """Sum of tensor elements over given axis."""
        from .functions.reductions import Sum
        return Sum(axis=axis, keepdims=keepdims)(self)
    
    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        """Mean of tensor elements over given axis."""
        from .functions.reductions import Mean
        return Mean(axis=axis, keepdims=keepdims)(self)
    
    def max(self, axis=None, keepdims=False) -> 'Tensor':
        """Maximum of tensor elements over given axis."""
        from .functions.reductions import Max
        return Max(axis=axis, keepdims=keepdims)(self)
    
    def min(self, axis=None, keepdims=False) -> 'Tensor':
        """Minimum of tensor elements over given axis."""
        from .functions.reductions import Min
        return Min(axis=axis, keepdims=keepdims)(self)
    
    def std(self, axis=None, keepdims=False, ddof=0) -> 'Tensor':
        """Standard deviation of tensor elements over given axis."""
        from .functions.reductions import Std
        return Std(axis=axis, keepdims=keepdims, ddof=ddof)(self)

    def argmin(self, axis=None) -> 'Tensor':
        """Indices of minimum values along an axis."""
        result = xp.argmin(self.data, axis=axis)
        return Tensor(result, requires_grad=False, name=self.name + "_argmin")

    def argmax(self, axis=None) -> 'Tensor':
        """Indices of maximum values along an axis."""
        result = xp.argmax(self.data, axis=axis)
        return Tensor(result, requires_grad=False, name=self.name + "_argmax")
    
    def log(self) -> 'Tensor':
        from .functions.math import Log
        return Log()(self)
    
    def exp(self) -> 'Tensor':
        from .functions.math import Exp
        return Exp()(self)
    
    def sin(self) -> 'Tensor':
        from .functions.math import Sin
        return Sin()(self)
    
    def cos(self) -> 'Tensor':
        from .functions.math import Cos
        return Cos()(self)
    
    def tan(self) -> 'Tensor':
        from .functions.math import Tan
        return Tan()(self)
    
    def sqrt(self) -> 'Tensor':
        from .functions.math import Sqrt
        return Sqrt()(self)
    
    def cbrt(self) -> 'Tensor':
        from .functions.math import Cbrt
        return Cbrt()(self)
    
    def log10(self) -> 'Tensor':
        from .functions.math import Log10
        return Log10()(self)
    
    def log2(self) -> 'Tensor':
        from .functions.math import Log2
        return Log2()(self)
    
    def abs(self) -> 'Tensor':
        from .functions.math import Abs
        return Abs()(self)
    
    def clip(self, min_val=None, max_val=None) -> 'Tensor':
        from .functions.math import Clip
        return Clip(min_val, max_val)(self)
    
    def relu(self) -> 'Tensor':
        from .functions.activations import ReLU
        return ReLU()(self)
    
    def sigmoid(self) -> 'Tensor':
        from .functions.activations import Sigmoid
        return Sigmoid()(self)
    
    def tanh(self) -> 'Tensor':
        from .functions.activations import Tanh
        return Tanh()(self)
    
    def leaky_relu(self, negative_slope: float = 0.01) -> 'Tensor':
        from .functions.activations import LeakyReLU
        return LeakyReLU(negative_slope=negative_slope)(self)
    
    def transpose(self, axes=None) -> 'Tensor':
        from .functions.linalg import Transpose
        return Transpose(axes=axes)(self)

    @property
    def T(self) -> 'Tensor':
        """Transpose of the tensor."""
        return self.transpose()
    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return self.data.shape
    @property
    def ndim(self) -> int:
        return self.data.ndim
    @property
    def size(self) -> int:
        return self.data.size
    @property
    def ndim(self) -> int:
        return self.data.ndim
     
    def __len__(self):
        return len(self.data)
    
    def reshape(self, new_shape: Union[int, Tuple[int, ...]]) -> 'Tensor':
        """Reshape the tensor to a new shape."""
        from .functions.tensor_ops import Reshape
        return Reshape(new_shape=new_shape)(self)

    def flatten(self) -> 'Tensor': # copy
        """Return a flattened 1D VIEW of the tensor, still attached to the computational graph."""
        from .functions.tensor_ops import Flatten
        return Flatten()(self)
    
    def squeeze(self, axis=None) -> 'Tensor':
        """Remove dimensions of size 1 from the tensor."""
        from .functions.tensor_ops import Squeeze
        return Squeeze(axis=axis)(self)

    def expand_dims(self, axis) -> 'Tensor':
        """Add new axis of size 1 at specified position"""
        from .functions.tensor_ops import ExpandDims
        return ExpandDims(axis=axis)(self)
    
    def pad(self, pad_width: Union[Sequence, ArrayLike, int], mode='constant', **kwargs) -> 'Tensor':
        """Pad the tensor with specified padding width and mode."""
        from .functions.tensor_ops import Pad
        return Pad(pad_width=pad_width, mode=mode, **kwargs)(self)

    def copy(self) -> 'Tensor':
        """Return a copy of the tensor."""
        copied_data = self.data.copy()
        new_tensor = Tensor(
            data=copied_data,
            requires_grad=self.requires_grad,
            grad_fn=None,  # Copy doesn't preserve gradient function
            name=self.name + "_copy",
            dtype=self.data.dtype
        )
        # Copy gradient if it exists
        if self.grad is not None:
            new_tensor.grad = self.grad.copy()
        return new_tensor

    def __getitem__(self, key):
        """Support slicing/indexing like a NumPy array while preserving metadata."""
        sliced_data = self.data[key]
        return self._new_tensor_like(sliced_data, name=self.name + f"_slice{key}")

    def __setitem__(self, key, value):
        """Allow setting values using index."""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def _new_tensor_like(self, data, name: Optional[str] = None) -> 'Tensor':
        """Create a new tensor with the same metadata as this one."""
        if not name:
            name = self.name + "_new"
        return Tensor(
            data=data,
            requires_grad=self.requires_grad,
            grad_fn=self.grad_fn,
            name=name,
            dtype=self.data.dtype,
        )
    
    def visualize_graph(self, **kwargs):
        """
        Visualize the computational graph that led to this tensor.
        
        Args:
            **kwargs: Additional arguments passed to the graph visualizer
            
        Returns:
            matplotlib Figure object
        """
        from .utils.graph import visualize_graph
        return visualize_graph(self, **kwargs)
    
    def save_graph(self, filename: str, **kwargs):
        """
        Save a visualization of the computational graph to file.
        
        Args:
            filename: Path to save the image
            **kwargs: Additional arguments passed to the graph visualizer
        """
        from .utils.graph import save_graph
        save_graph(self, filename, **kwargs)
    
    def print_graph(self):
        """
        Print a text representation of the computational graph structure.
        """
        from .utils.graph import print_graph_structure
        print_graph_structure(self)
    
    def graph_stats(self):
        """
        Get statistics about the computational graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        from .utils.graph import get_graph_stats
        return get_graph_stats(self)
    
    def __hash__(self):
        """Make tensor hashable using its id."""
        return hash(id(self))
    
    def __repr__(self):
        """Return a string representation of the tensor."""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn}, name={self.name})"

    def __str__(self):
        """Return a string representation of the tensor."""
        return f"Tensor({self.data})"


def zeros(shape: Union[int, List[int]], dtype: Optional[str] = None,
          requires_grad: bool = False) -> Tensor:
    return Tensor(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def ones(shape: Union[int, List[int]], dtype: Optional[str] = None,
         requires_grad: bool = False) -> Tensor:
    return Tensor(xp.ones(shape, dtype=dtype), requires_grad=requires_grad)

def zeros_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(xp.zeros_like(tensor.data, dtype=tensor.data.dtype), requires_grad=requires_grad, name=tensor.name + "_zeros_like")

def ones_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(xp.ones_like(tensor.data, dtype=tensor.data.dtype), requires_grad=requires_grad, name=tensor.name + "_ones_like")

def empty(shape: Union[int, List[int]], dtype: Optional[str] = None,
          requires_grad: bool = False) -> Tensor:
    return Tensor(xp.empty(shape, dtype=dtype), requires_grad=requires_grad)

def arange(start: int, stop: int, step: int = 1, dtype: Optional[str] = None,
           requires_grad: bool = False) -> Tensor:
    return Tensor(xp.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)

def eye(n: int, dtype: Optional[str] = None,
        requires_grad: bool = False) -> Tensor:
    return Tensor(xp.eye(n, dtype=dtype), requires_grad=requires_grad)

def argmin(tensor: Tensor, axis: Optional[int] = None) -> Tensor:
    """Return indices of minimum values along an axis."""
    return tensor.argmin(axis=axis)

def argmax(tensor: Tensor, axis: Optional[int] = None) -> Tensor:
    """Return indices of maximum values along an axis."""
    return tensor.argmax(axis=axis)