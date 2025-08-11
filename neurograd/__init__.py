# Device detection and numpy/cupy setup must happen first to avoid circular imports
from .utils.device import auto_detect_device
DEVICE = auto_detect_device()
if DEVICE == "cpu":
    import numpy as xp
elif DEVICE == "cuda":
    import cupy as xp

# Now import everything else after xp is available
from .functions import (arithmetic, math, linalg, activations, reductions, conv)
from .functions.arithmetic import add, sub, mul, div, pow
from .functions.math import log, exp, sin, cos, tan, sqrt, cbrt, log10, log2, abs, clip
from .functions.linalg import matmul, dot, tensordot, transpose
from .functions.tensor_ops import reshape, flatten, squeeze, expand_dims, cast, pad, sliding_window_view, newaxis
from .functions.reductions import Sum, Mean, Max, Min, Std, sum, mean, max, min, std
from .functions.conv import conv2d, pool2d, maxpool2d, averagepool2d, pooling2d, maxpooling2d, averagepooling2d
from .tensor import Tensor, ones, zeros, ones_like, zeros_like, empty, arange, eye

# Automatic Mixed Precision (AMP) support
try:
    from .amp import autocast, GradScaler
except ImportError:
    # Define dummy functions if AMP module not available
    def autocast(*args, **kwargs):
        import contextlib
        return contextlib.nullcontext()
    
    class GradScaler:
        def __init__(self, *args, **kwargs):
            pass
        def scale(self, x):
            return x
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
# Optional graph visualization (requires matplotlib)
try:
    from .utils.graph import visualize_graph, save_graph, print_graph_structure
except ImportError:
    # Define dummy functions if matplotlib is not available
    def visualize_graph(*args, **kwargs):
        print("Graph visualization requires matplotlib")
    def save_graph(*args, **kwargs):
        print("Graph saving requires matplotlib")
    def print_graph_structure(*args, **kwargs):
        print("Graph structure printing requires matplotlib")



# Importing numpy data types for convenience. This allows users to use float32, int64, etc. directly
for name in ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool_']:
    globals()[name] = getattr(xp, name)


def save(obj, f, protocol=None):
    """Serialize with cloudpickle if available."""
    try:
        import cloudpickle as _p
    except Exception:
        import pickle as _p
    import pickle as _std
    protocol = _std.HIGHEST_PROTOCOL if protocol is None else protocol
    if isinstance(f, (str, bytes)):
        with open(f, "wb") as fh:
            _p.dump(obj, fh, protocol=protocol)
    else:
        _p.dump(obj, f, protocol=protocol)


def load(f):
    """Deserialize with cloudpickle if available."""
    try:
        import cloudpickle as _p
    except Exception:
        import pickle as _p
    if isinstance(f, (str, bytes)):
        with open(f, "rb") as fh:
            return _p.load(fh)
    return _p.load(f)
