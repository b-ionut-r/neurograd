# Device detection and numpy/cupy setup must happen first to avoid circular imports
from .utils.device import auto_detect_device
DEVICE = auto_detect_device()
if DEVICE == "cpu":
    import numpy as xp
elif DEVICE == "cuda":
    import cupy as xp

# Now import everything else after xp is available
from .functions import (arithmetic, math, linalg, activations, reductions)
from .functions.arithmetic import add, sub, mul, div, pow
from .functions.math import log, exp, sin, cos, tan, sqrt, cbrt, log10, log2, abs
from .functions.linalg import matmul, dot, transpose
from .functions.reductions import Sum, Mean, Max, Min, Std, sum, mean, max, min, std
from .tensor import Tensor, ones, zeros, ones_like, zeros_like, arange
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