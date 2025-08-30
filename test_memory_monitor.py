"""
Quick manual test for the per-op MemoryMonitor.

Runs a tiny forward/backward pass and prints per-op memory snapshots when
running on CUDA (CuPy). On CPU, only op names/shapes are printed.

Usage:
  python -u scripts/test_memory_monitor.py
"""

import time

import neurograd as ng
from neurograd import Tensor, xp
from neurograd.utils.memory import MemoryMonitor, log_point


def main():
    print(f"NeuroGrad DEVICE={ng.DEVICE}")
    # Small shapes so it runs anywhere
    N, C, H, W = 8, 3, 32, 32
    K = 10
    dtype = xp.float16 if ng.DEVICE == "cuda" else xp.float32

    # Dummy data
    X = Tensor(xp.random.randn(N, C, H, W).astype(dtype))
    y = Tensor(xp.eye(K, dtype=xp.float32)[xp.random.randint(0, K, size=(N,))])

    from neurograd.nn.layers import Conv2D, MaxPool2D, Flatten, Linear, Sequential
    from neurograd.nn.losses import CategoricalCrossEntropy
    from neurograd.optim import SGD

    model = Sequential(
        Conv2D(3, 8, 3, padding="same", activation="relu"),
        MaxPool2D(2, strides=2),
        Flatten(),
        Linear(8 * 16 * 16, 32, activation="relu"),
        Linear(32, K),
    )
    opt = SGD(model.named_parameters(), lr=1e-2)
    loss_fn = CategoricalCrossEntropy(from_logits=True)

    # Enable per-op logging for a single step
    with MemoryMonitor(prefix="[VRAM]", include_driver=True, include_pool=True, include_fft=False):
        t0 = time.time()
        log_point("before forward")
        out = model(X)
        loss = loss_fn(y, out)
        log_point("before backward")
        loss.backward()
        log_point("before step")
        opt.step()
        dt = (time.time() - t0) * 1e3
        log_point(f"after step ({dt:.1f} ms)")

    print("Done.")


if __name__ == "__main__":
    main()

