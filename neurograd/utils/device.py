import numpy as real_np

def auto_detect_device():
    try:
        import cupy as cp
        _ = cp.zeros(1)  # Will fail if driver is insufficient
        return 'cuda'
    except Exception:  # Catch more than just ImportError
        return 'cpu'

def array_to_numpy(x):
    if isinstance(x, real_np):
        return x
    else:
        return x.asnumpy()