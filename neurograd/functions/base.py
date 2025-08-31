from abc import ABC, abstractmethod
from typing import List, Tuple
import neurograd as ng
from neurograd.tensor import Tensor
from neurograd import xp
from neurograd.utils.memory import maybe_log_op_memory


class Function(ABC):
    name = None
    def __init__(self):
        self.parent_tensors: List[Tensor] = []

    def __call__(self, *inputs) -> Tensor:
        processed_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, Tensor):
                processed_inputs.append(inp)
            else:
                try:
                    data = xp.array(inp)
                    processed_inputs.append(Tensor(data, requires_grad=False))
                except Exception as e:
                    raise TypeError(f"Input {i} must be convertible to numpy array, got {type(inp)}") from e
        
        # Apply autocast if enabled (but not for Cast operations to avoid recursion)
        try:
            from neurograd.amp.autocast import is_autocast_enabled
            if is_autocast_enabled():
                from neurograd.amp.utils import maybe_cast_tensor
                op_name = getattr(self, 'name', None) or self.__class__.__name__
                if op_name != 'Cast':  # Avoid recursion with Cast operations
                    processed_inputs = [maybe_cast_tensor(inp, op_name=op_name) for inp in processed_inputs]
        except ImportError:
            # AMP not available, continue with original precision
            pass
        
        self.parent_tensors = processed_inputs
        output_data = self.forward(*[inp.data for inp in processed_inputs])
        requires_grad = any(inp.requires_grad for inp in processed_inputs)
        output = Tensor(output_data, requires_grad=requires_grad, grad_fn=self)
        # Do memsave if ops has one
        if hasattr(self, "_memsave") and getattr(self, "memsave", False):
            self._memsave(self.parent_tensors, output)
        # Optional per-op memory logging (enabled only inside MemoryMonitor)
        try:
            op_name = getattr(self, 'name', None) or self.__class__.__name__
            maybe_log_op_memory(op_name, self.parent_tensors, output_data)
        except Exception:
            pass
        return output
    
    @abstractmethod
    def forward(self, *inputs: xp.ndarray) -> xp.ndarray:
        """
        Forward pass of the function.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: xp.ndarray) -> Tuple[xp.ndarray, ...]:
        """
        Backward pass of the function.
        Must be implemented by subclasses.
        Returns gradients with respect to inputs.
        """
        pass


    def _handle_broadcasting(self, grad: xp.ndarray, original_shape: tuple) -> xp.ndarray:
        """
        Handle broadcasting by summing gradients over broadcasted dimensions.
        
        Args:
            grad: The gradient tensor that may have been broadcasted
            original_shape: The original shape of the tensor before broadcasting
            
        Returns:
            Gradient tensor with shape matching original_shape
        """
        if grad is None:
            return None
            
        # Sum over dimensions that were added during broadcasting
        while grad.ndim > len(original_shape):
            grad = xp.sum(grad, axis=0) # sum
        
        # Sum over dimensions that were broadcasted (size 1 -> size N)
        for i in range(len(original_shape)):
            if original_shape[i] == 1 and grad.shape[i] > 1:
                grad = xp.sum(grad, axis=i, keepdims=True)
        
        return grad
