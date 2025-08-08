"""
Utility functions for automatic mixed precision

This module contains utility functions used by the AMP system for
determining precision, casting tensors, and managing operation types.
"""

from typing import Set
from .autocast import autocast


# Set of operations that should always stay in FP32 for numerical stability  
_FP32_OPS: Set[str] = {
    'exp', 'log', 'sqrt', 'softmax', 'log_softmax', 'layer_norm', 'batchnorm',
    'batchnorm2d',
    'cross_entropy', 'mse_loss', 'l1_loss', 'smooth_l1_loss', 'binarycrossentropy',
    'sum', 'mean', 'std', 'var', 'norm', 'categoricalcrossentropy', 'mse', 'cast'
}

# Set of operations that are safe to run in FP16
_FP16_SAFE_OPS: Set[str] = {
    'add', 'sub', 'mul', 'div', 'dot', 'matmul', 'conv2d', 'relu', 'gelu', 'tanh', 'sigmoid',
    'max', 'min', 'transpose', 'reshape', 'flatten', 'linear'
}


def should_cast_to_fp16(op_name: str) -> bool:
    """
    Determine if an operation should be cast to FP16 in autocast context.
    
    Args:
        op_name: Name of the operation
        
    Returns:
        True if the operation should be cast to FP16, False otherwise
    """
    if not autocast.is_enabled():
        return False
    
    # Handle None or empty op_name
    if not op_name:
        return True  # Default to allowing FP16
    
    op_name_lower = op_name.lower()
    
    # Force certain ops to stay in FP32
    if op_name_lower in _FP32_OPS:
        return False
        
    # Allow safe ops to use FP16
    if op_name_lower in _FP16_SAFE_OPS:
        return True
        
    # Default behavior: allow FP16 for most ops but with caution
    return True


def maybe_cast_tensor(tensor, target_dtype=None, op_name: str = "unknown") -> 'Tensor':
    """
    Cast tensor to appropriate dtype based on autocast context and operation type.
    
    Args:
        tensor: Input tensor
        target_dtype: Target dtype (if None, uses autocast dtype)
        op_name: Name of the operation for casting decision
        
    Returns:
        Tensor cast to appropriate dtype
    """
    from neurograd.tensor import Tensor
    
    if not isinstance(tensor, Tensor):
        return tensor
        
    if not autocast.is_enabled():
        return tensor
    
    # Determine target dtype
    if target_dtype is None:
        if should_cast_to_fp16(op_name):
            target_dtype = autocast.get_autocast_dtype()
        else:
            # Keep in original precision for sensitive ops
            return tensor
    
    # Only cast if different from current dtype
    if tensor.data.dtype == target_dtype:
        return tensor
        
    return tensor.cast(target_dtype)


def get_fp32_ops() -> Set[str]:
    """Get the set of operations that should stay in FP32."""
    return _FP32_OPS.copy()


def get_fp16_safe_ops() -> Set[str]:
    """Get the set of operations that are safe for FP16."""
    return _FP16_SAFE_OPS.copy()


def add_fp32_op(op_name: str) -> None:
    """Add an operation to the FP32 operations set."""
    _FP32_OPS.add(op_name.lower())


def add_fp16_safe_op(op_name: str) -> None:
    """Add an operation to the FP16-safe operations set."""
    _FP16_SAFE_OPS.add(op_name.lower())


def remove_fp32_op(op_name: str) -> None:
    """Remove an operation from the FP32 operations set."""
    _FP32_OPS.discard(op_name.lower())


def remove_fp16_safe_op(op_name: str) -> None:
    """Remove an operation from the FP16-safe operations set."""
    _FP16_SAFE_OPS.discard(op_name.lower())