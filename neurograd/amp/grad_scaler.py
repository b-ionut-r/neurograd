"""
Gradient scaler for automatic mixed precision training

This module provides the GradScaler class that handles gradient scaling
to prevent underflow when training with FP16 gradients.
"""

from typing import Any, Dict
from neurograd import xp
import numpy as real_np


class GradScaler:
    """
    Gradient scaler for automatic mixed precision training.
    
    When training with FP16, gradients can underflow (become zero) due to the limited
    numerical range of FP16. GradScaler addresses this by scaling the loss before
    backward pass and unscaling gradients before optimizer step.
    
    Example:
        >>> scaler = GradScaler()
        >>> for inputs, targets in dataloader:
        ...     optimizer.zero_grad()
        ...     with autocast():
        ...         outputs = model(inputs)
        ...         loss = loss_fn(outputs, targets)
        ...     
        ...     # Scale loss to prevent gradient underflow
        ...     scaled_loss = scaler.scale(loss)
        ...     scaled_loss.backward()
        ...     
        ...     # Unscale gradients and step
        ...     scaler.step(optimizer)
        ...     scaler.update()
    """
    
    def __init__(self, 
                 init_scale: float = 2**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        """
        Initialize gradient scaler.
        
        Args:
            init_scale: Initial scaling factor
            growth_factor: Factor by which to multiply the scale when no overflow is detected
            backoff_factor: Factor by which to multiply the scale when overflow is detected  
            growth_interval: Number of steps between scale increases
            enabled: Whether gradient scaling is enabled
        """
        if not enabled:
            # Disable scaling - useful for debugging or CPU training
            self._enabled = False
            return
            
        self._enabled = True
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        
        # Tracking variables
        self._growth_tracker = 0
        self._found_inf = False
        
    def is_enabled(self) -> bool:
        """Check if gradient scaling is enabled."""
        return self._enabled
    
    def get_scale(self) -> float:
        """Get current scale factor."""
        return self._scale if self._enabled else 1.0
    
    def scale(self, tensor) -> 'Tensor':
        """
        Scale a tensor (typically the loss) by the current scale factor.
        
        Args:
            tensor: Tensor to scale (usually the loss)
            
        Returns:
            Scaled tensor
        """
        if not self._enabled:
            return tensor
            
        from neurograd.tensor import Tensor
        
        if not isinstance(tensor, Tensor):
            # Convert to tensor if needed
            tensor = Tensor(xp.array(tensor), requires_grad=tensor.requires_grad if hasattr(tensor, 'requires_grad') else False)
        
        # Scale by multiplying with scale factor
        scale_tensor = Tensor(xp.array(self._scale), requires_grad=False)
        return tensor * scale_tensor
    
    def unscale_(self, optimizer) -> None:
        """
        Unscale gradients in-place for the given optimizer.
        
        This should be called before optimizer.step() to restore gradients to their
        original magnitude.
        
        Args:
            optimizer: Optimizer whose parameters' gradients should be unscaled
        """
        if not self._enabled:
            return
            
        # Check for inf/nan gradients and unscale
        found_inf = False
        
        for param_name, param in optimizer.params:
            if param.requires_grad and param.grad is not None:
                # Check for inf/nan
                grad_data = param.grad
                if hasattr(grad_data, 'get'):  # CuPy array
                    grad_cpu = grad_data.get()
                else:
                    grad_cpu = grad_data
                
                if not real_np.isfinite(grad_cpu).all():
                    found_inf = True
                    break
                
                # Unscale gradient (avoid division by zero)
                if self._scale > 0:
                    param.grad = param.grad / self._scale
        
        self._found_inf = found_inf
    
    def step(self, optimizer) -> None:
        """
        Perform optimizer step with gradient scaling.
        
        This unscales gradients, checks for overflow, and conditionally runs
        the optimizer step.
        
        Args:
            optimizer: Optimizer to step
        """
        if not self._enabled:
            optimizer.step()
            return
        
        # Unscale gradients
        self.unscale_(optimizer)
        
        # Only step if no overflow detected
        if not self._found_inf:
            optimizer.step()
    
    def update(self) -> None:
        """
        Update the scale factor based on recent gradient overflow status.
        
        Should be called after each training step to adjust the scale factor
        for the next iteration.
        """
        if not self._enabled:
            return
        
        if self._found_inf:
            # Overflow detected - reduce scale
            self._scale *= self._backoff_factor
            # Prevent scale from going too low
            self._scale = max(self._scale, 1.0)
            self._growth_tracker = 0
            self._found_inf = False
        else:
            # No overflow - consider growing scale
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            'found_inf': self._found_inf,
            'growth_factor': self._growth_factor,
            'backoff_factor': self._backoff_factor,
            'growth_interval': self._growth_interval,
            'enabled': self._enabled
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary from checkpoint."""
        self._scale = state_dict['scale']
        self._growth_tracker = state_dict['growth_tracker'] 
        self._found_inf = state_dict['found_inf']
        self._growth_factor = state_dict['growth_factor']
        self._backoff_factor = state_dict['backoff_factor']
        self._growth_interval = state_dict['growth_interval']
        self._enabled = state_dict['enabled']