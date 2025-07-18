def auto_detect_device():
    """Check if CUDA is available."""
    try:
        import cupy as cp
        _ = cp.zeros(1)  # Test if cupy can create a tensor
        return 'cuda'
    except ImportError:
        return 'cpu'