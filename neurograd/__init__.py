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
from .functions.tensor_ops import reshape, flatten, squeeze, expand_dims, concat, cast, pad, sliding_window_view, newaxis
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


def save(obj, f, protocol=None, portable: bool = True):
    """
    Serialize an object with cloudpickle if available.

    If portable=True, convert CuPy arrays and NeuroGrad Tensors to NumPy first so
    the checkpoint can be loaded on CPU-only environments without importing CuPy.
    """

    # Lightweight recursive conversion to NumPy where needed
    def _to_portable(x):
        if not portable:
            return x
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None
        try:
            import cupy as _cp  # type: ignore
        except Exception:
            _cp = None

        # NeuroGrad Tensor -> NumPy array
        try:
            from .tensor import Tensor as _Tensor  # local import to avoid cycles
            if isinstance(x, _Tensor):
                return _np.asarray(x.data) if _np is not None else x.data
        except Exception:
            pass

        # CuPy array -> NumPy array
        if _cp is not None:
            try:
                if isinstance(x, _cp.ndarray):  # type: ignore[attr-defined]
                    return _cp.asnumpy(x)
            except Exception:
                pass

        # NumPy array -> ensure plain ndarray (no memmaps, etc.)
        if _np is not None:
            try:
                if isinstance(x, _np.ndarray):
                    return _np.asarray(x)
            except Exception:
                pass

        # Containers
        if isinstance(x, dict):
            return {k: _to_portable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = type(x)
            return t(_to_portable(v) for v in x)
        if isinstance(x, set):
            return { _to_portable(v) for v in x }

        return x

    try:
        import cloudpickle as _p
    except Exception:
        import pickle as _p
    import pickle as _std
    protocol = _std.HIGHEST_PROTOCOL if protocol is None else protocol
    obj_to_save = _to_portable(obj)
    if isinstance(f, (str, bytes)):
        with open(f, "wb") as fh:
            _p.dump(obj_to_save, fh, protocol=protocol)
    else:
        _p.dump(obj_to_save, f, protocol=protocol)


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
    
