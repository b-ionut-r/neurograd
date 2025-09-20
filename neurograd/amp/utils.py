"""
Utility functions for automatic mixed precision

This module contains utility functions used by the AMP system for
determining precision, casting tensors, and managing operation types.
"""
import neurograd as ng
from typing import Set
from .autocast import autocast


# Ops that should **always** run in FP32 for numerical stability
_FP32_OPS: Set[str] = {
    # math with steep/ill-conditioned curves
    "log", "exp", "sqrt", "cbrt", "log10", "log2",
    "sin", "cos", "tan",
    # softmax & reductions that accumulate / are variance-based
    "softmax",
    "sum", "mean", "std", "var",
    # losses (compute entirely in fp32)
    "mse", "rmse", "mae", "binarycrossentropy", "categoricalcrossentropy",
    # casting decisions shouldn’t be overridden by autocast
    "cast",
    # pow is risky in fp16 except for tiny integer exponents
    "pow",
    # normalization
    "batchnormalizer", "batchnormalizercudnn",
}

# Ops that are **safe** to run in FP16 (or BF16) by default
_FP16_SAFE_OPS: Set[str] = {
    # arithmetic
    "add", "sub", "mul", "div",
    # linalg (prefer fp32 accumulation inside the kernel)
    "matmul", "tensordot", "einsum", "transpose", "convolver",
    # tensor reshapes / views
    "reshape", "flatten", "squeeze", "expanddims", "slidingwindowview",
    # padding and elementwise
    "pad", "abs", "clip", "max", "min",
    # activations (excluding softmax)
    "relu", "relu6", "leakyrelu", "sigmoid", "tanh", "passthrough",
}



# FP32 ops that should not pre-cast inputs; instead, compute with FP32 accumulation
_FP32_NO_PRECAST: Set[str] = {"sum", "mean", "std", "var", "batchnormalizer",
                              "batchnormalizercudnn"}

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
            # Avoid pre-casting large tensors for certain FP32 ops; rely on op to use FP32 accumulate
            if op_name and op_name.lower() in _FP32_NO_PRECAST:
                return tensor
            return tensor.cast(ng.float32)
    
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
