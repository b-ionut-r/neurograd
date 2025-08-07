"""
NeuroGrad Automatic Mixed Precision (AMP) Module

This module provides automatic mixed precision training capabilities for NeuroGrad,
enabling faster training and reduced memory usage while maintaining numerical stability.

The module is organized into several components:
- autocast: Context manager for automatic precision casting
- grad_scaler: Gradient scaling for FP16 training stability
- utils: Utility functions for mixed precision operations

Example usage:
    >>> from neurograd.amp import autocast, GradScaler
    >>> 
    >>> scaler = GradScaler()
    >>> for inputs, targets in dataloader:
    ...     optimizer.zero_grad()
    ...     with autocast():
    ...         outputs = model(inputs)
    ...         loss = loss_fn(outputs, targets)
    ...     
    ...     scaled_loss = scaler.scale(loss)
    ...     scaled_loss.backward()
    ...     scaler.step(optimizer)
    ...     scaler.update()
"""

from .autocast import autocast, is_autocast_enabled, get_autocast_dtype
from .grad_scaler import GradScaler
from .utils import should_cast_to_fp16, maybe_cast_tensor

# For backward compatibility and convenience
__all__ = [
    'autocast',
    'GradScaler', 
    'is_autocast_enabled',
    'get_autocast_dtype',
    'should_cast_to_fp16',
    'maybe_cast_tensor'
]