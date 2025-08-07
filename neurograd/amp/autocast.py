"""
Autocast context manager for automatic mixed precision

This module provides the autocast context manager that automatically selects
the appropriate precision (FP16 vs FP32) for tensor operations based on
their numerical stability requirements.
"""

import threading
from typing import Optional
from neurograd import xp


class autocast:
    """
    Context manager that enables automatic mixed precision training.
    
    In the autocast context, tensor operations are automatically cast to the appropriate
    precision: FP16 for most forward pass operations to speed up computation, and FP32 
    for operations that require higher precision for numerical stability.
    
    Example:
        >>> with autocast(enabled=True):
        ...     output = model(input)
        ...     loss = loss_fn(target, output)  # Loss computation stays in FP32
    """
    
    # Thread-local storage for autocast state
    _local = threading.local()
    
    def __init__(self, device_type: str = "cuda", enabled: bool = True, dtype: Optional[str] = None):
        """
        Initialize autocast context manager.
        
        Args:
            device_type: Device type ('cuda' or 'cpu'). Currently only 'cuda' benefits from mixed precision.
            enabled: Whether to enable autocast. If False, operations run in their original precision.
            dtype: Target dtype for autocast. If None, uses float16 for CUDA, float32 for CPU.
        """
        if device_type not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device type: {device_type}")
        
        self.device_type = device_type
        self.enabled = enabled
        
        # Determine target dtype based on device and user preference
        if dtype is None:
            self.dtype = xp.float16 if device_type == "cuda" and enabled else xp.float32
        else:
            if isinstance(dtype, str):
                self.dtype = getattr(xp, dtype)
            else:
                self.dtype = dtype
                
        self.prev_enabled = None
        self.prev_dtype = None

    def __enter__(self):
        """Enter the autocast context."""
        # Save previous state
        self.prev_enabled = getattr(autocast._local, 'enabled', False)
        self.prev_dtype = getattr(autocast._local, 'dtype', xp.float32)
        
        # Set new state
        autocast._local.enabled = self.enabled
        autocast._local.dtype = self.dtype
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the autocast context."""
        # Restore previous state
        autocast._local.enabled = self.prev_enabled
        autocast._local.dtype = self.prev_dtype

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if autocast is currently enabled."""
        return getattr(cls._local, 'enabled', False)
    
    @classmethod
    def get_autocast_dtype(cls):
        """Get the current autocast dtype."""
        return getattr(cls._local, 'dtype', xp.float32)


def is_autocast_enabled() -> bool:
    """Check if autocast is currently enabled (convenience function)."""
    return autocast.is_enabled()


def get_autocast_dtype():
    """Get current autocast dtype (convenience function).""" 
    return autocast.get_autocast_dtype()