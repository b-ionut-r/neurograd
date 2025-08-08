"""
Gradient scaler for automatic mixed precision training

This module provides the GradScaler class that handles gradient scaling
to prevent underflow when training with FP16 gradients.
"""

from typing import Any, Dict, Optional, List, Tuple
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
                 init_scale: float = 256.0,      # More conservative starting point
                 growth_factor: float = 2.0,     # Standard doubling
                 backoff_factor: float = 0.5,    # Standard halving
                 growth_interval: int = 2000,    # Wait 2k steps before growing
                 max_scale: float = 2**20,       # Cap at ~1M to prevent overflow
                 min_scale: float = 1.0,         # Don't go below 1.0
                 enabled: bool = True):
        """
        Initialize gradient scaler with best practice defaults.
        
        Args:
            init_scale: Initial scaling factor (default: 256.0)
                - 256 is a good balance: high enough to prevent underflow,
                  low enough to avoid immediate overflow
            growth_factor: Factor by which to multiply the scale when no overflow 
                is detected (default: 2.0)
            backoff_factor: Factor by which to multiply the scale when overflow 
                is detected (default: 0.5)
            growth_interval: Number of consecutive steps without overflow before
                scale is increased (default: 2000)
            max_scale: Maximum allowed scale value to prevent unbounded growth
                (default: 2^20 = 1,048,576)
            min_scale: Minimum allowed scale value (default: 1.0)
            enabled: Whether gradient scaling is enabled (default: True)
        """
        self._enabled = enabled
        
        if not enabled:
            # When disabled, all operations become no-ops
            return
        
        # Validate inputs
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {growth_factor}")
        if backoff_factor <= 0 or backoff_factor >= 1.0:
            raise ValueError(f"backoff_factor must be in (0, 1), got {backoff_factor}")
        if growth_interval < 1:
            raise ValueError(f"growth_interval must be >= 1, got {growth_interval}")
        if max_scale < init_scale:
            raise ValueError(f"max_scale must be >= init_scale")
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}")
            
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._max_scale = float(max_scale)
        self._min_scale = float(min_scale)
        
        # Tracking variables
        self._growth_tracker = 0         # Steps since last scale change
        self._found_inf_this_step = False  # Did we find inf/nan this step?
        self._has_been_unscaled = False  # Prevent double unscaling
        
        # Statistics tracking (useful for debugging)
        self._num_overflows = 0
        self._num_scale_ups = 0
        self._num_scale_downs = 0
        
    def is_enabled(self) -> bool:
        """Check if gradient scaling is enabled."""
        return self._enabled
    
    def get_scale(self) -> float:
        """Get current scale factor."""
        return self._scale if self._enabled else 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scaling statistics for debugging/monitoring."""
        if not self._enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "current_scale": self._scale,
            "num_overflows": self._num_overflows,
            "num_scale_ups": self._num_scale_ups,
            "num_scale_downs": self._num_scale_downs,
            "growth_tracker": self._growth_tracker
        }
    
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
            tensor = Tensor(
                xp.array(tensor), 
                requires_grad=tensor.requires_grad if hasattr(tensor, 'requires_grad') else False
            )
        
        # Scale by multiplying with scale factor
        scale_tensor = Tensor(xp.array(self._scale), requires_grad=False)
        return tensor * scale_tensor
    
    def unscale_(self, optimizer) -> None:
        """
        Unscale gradients in-place for the given optimizer.
        
        This should be called before optimizer.step() to restore gradients to their
        original magnitude. Checks for inf/nan AFTER unscaling.
        
        Args:
            optimizer: Optimizer whose parameters' gradients should be unscaled
        """
        if not self._enabled:
            return
        
        # Prevent double unscaling in the same step
        if self._has_been_unscaled:
            return
            
        # Calculate inverse scale once
        inv_scale = 1.0 / self._scale if self._scale > 0 else 0.0
        found_inf = False
        
        # Process all parameters
        for param_name, param in optimizer.params:
            if param.requires_grad and param.grad is not None:
                # Unscale gradient FIRST
                param.grad = param.grad * inv_scale
                
                # THEN check the UNSCALED gradient for inf/nan
                grad_data = param.grad
                
                # Handle both CuPy and NumPy arrays
                if hasattr(grad_data, 'get'):  # CuPy array
                    grad_cpu = grad_data.get()
                else:
                    grad_cpu = grad_data
                
                # Check for inf/nan in unscaled gradient
                if not real_np.isfinite(grad_cpu).all():
                    found_inf = True
                    # Zero out inf/nan gradients to prevent optimizer issues
                    # This prevents the optimizer from updating with bad values
                    param.grad = param.grad * 0
        
        self._found_inf_this_step = found_inf
        self._has_been_unscaled = True
        
        # Track statistics
        if found_inf:
            self._num_overflows += 1
    
    def step(self, optimizer) -> bool:
        """
        Perform optimizer step with gradient scaling.
        
        This unscales gradients, checks for overflow, and conditionally runs
        the optimizer step.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            True if optimizer step was taken, False if skipped due to overflow
        """
        if not self._enabled:
            optimizer.step()
            return True
        
        # Unscale gradients if not already done
        if not self._has_been_unscaled:
            self.unscale_(optimizer)
        
        # Only step if no overflow detected
        if not self._found_inf_this_step:
            optimizer.step()
            return True
        else:
            return False
    
    def update(self, new_scale: Optional[float] = None) -> None:
        """
        Update the scale factor based on recent gradient overflow status.
        
        Should be called after each training step to adjust the scale factor
        for the next iteration.
        
        Args:
            new_scale: If provided, sets the scale to this value instead of
                      using automatic adjustment
        """
        if not self._enabled:
            return
        
        # Allow manual scale override
        if new_scale is not None:
            self._scale = float(new_scale)
            self._scale = max(self._min_scale, min(self._scale, self._max_scale))
            self._growth_tracker = 0
        elif self._found_inf_this_step:
            # Overflow detected - reduce scale
            self._scale *= self._backoff_factor
            self._scale = max(self._scale, self._min_scale)
            self._growth_tracker = 0
            self._num_scale_downs += 1
        else:
            # No overflow - consider growing scale
            self._growth_tracker += 1
            
            if self._growth_tracker >= self._growth_interval:
                old_scale = self._scale
                self._scale *= self._growth_factor
                # Clamp to maximum scale to prevent unbounded growth
                self._scale = min(self._scale, self._max_scale)
                
                # Only count as scale up if we actually increased
                if self._scale > old_scale:
                    self._num_scale_ups += 1
                    
                self._growth_tracker = 0
        
        # Reset per-step flags
        self._found_inf_this_step = False
        self._has_been_unscaled = False
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            Dictionary containing all state needed to resume training
        """
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            'growth_factor': self._growth_factor,
            'backoff_factor': self._backoff_factor,
            'growth_interval': self._growth_interval,
            'max_scale': self._max_scale,
            'min_scale': self._min_scale,
            'enabled': self._enabled,
            # Statistics
            'num_overflows': self._num_overflows,
            'num_scale_ups': self._num_scale_ups,
            'num_scale_downs': self._num_scale_downs
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state dictionary from checkpoint.
        
        Args:
            state_dict: State dictionary from state_dict() call
        """
        self._scale = state_dict['scale']
        self._growth_tracker = state_dict['growth_tracker']
        self._growth_factor = state_dict['growth_factor']
        self._backoff_factor = state_dict['backoff_factor']
        self._growth_interval = state_dict['growth_interval']
        self._enabled = state_dict['enabled']
        
        # Load bounds if present (for backward compatibility)
        self._max_scale = state_dict.get('max_scale', 2**20)
        self._min_scale = state_dict.get('min_scale', 1.0)
        
        # Load statistics if present
        self._num_overflows = state_dict.get('num_overflows', 0)
        self._num_scale_ups = state_dict.get('num_scale_ups', 0)
        self._num_scale_downs = state_dict.get('num_scale_downs', 0)
        
        # Reset per-step flags
        self._found_inf_this_step = False
        self._has_been_unscaled = False
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if not self._enabled:
            return "GradScaler(enabled=False)"
        
        return (
            f"GradScaler(scale={self._scale:.1f}, "
            f"growth_factor={self._growth_factor}, "
            f"backoff_factor={self._backoff_factor}, "
            f"growth_interval={self._growth_interval}, "
            f"overflows={self._num_overflows})"
        )