# intensive_profile.py
# Make NeuroGrad training heavier for Scalene GPU+memory profiling,
# while staying under ≈4 GB VRAM (GTX 1650 Ti). Dynamic OOM backoff included.

import os, time, math, contextlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====== NeuroGrad imports ======
import neurograd as ng
from neurograd import Tensor
from neurograd.nn.layers.conv import Conv2D, MaxPool2D
from neurograd.nn.layers.linear import Linear
from neurograd.nn.module import Sequential
from neurograd.nn.losses import CategoricalCrossEntropy
from neurograd.functions.activations import ReLU, Softmax
from neurograd.optim.adam import Adam
from neurograd.utils.data import Dataset, DataLoader
from neurograd.nn.metrics import accuracy_score as ng_accuracy_score
# Flatten (kept like your code)
from neurograd.nn.layers import Flatten

# ====== Optional AMP ======
USE_MIXED_PRECISION = True   # Set False for higher memory use
try:
    from neurograd.amp import autocast, GradScaler
except Exception:
    USE_MIXED_PRECISION = False
    autocast = contextlib.nullcontext
    GradScaler = None

# ====== Style & seed ======
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print(f"NeuroGrad device: {ng.DEVICE} | Backend: {'CuPy' if ng.DEVICE=='cuda' else 'NumPy'}")
print(f"Mixed precision: {'ON' if USE_MIXED_PRECISION else 'OFF'}")

# ====== GPU helpers ======
def gpu_mem_gb():
    """Return (used_GB, total_GB) if CUDA, else (0,0)."""
    if ng.DEVICE != "cuda":
        return 0.0, 0.0
    import cupy as cp
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    used = (total_b - free_b) / (1024**3)
    total = total_b / (1024**3)
    return used, total

def gpu_sync():
    if ng.DEVICE == "cuda":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

def log_vram(tag):
    used, total = gpu_mem_gb()
    if total > 0:
        print(f"[VRAM] {tag}: used {used:.2f} GB / {total:.2f} GB")

# ====== Load and inflate dataset ======
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target   # (1797, 64)

# Reshape to NCHW (N,1,8,8)
X_images = X.reshape(-1, 8, 8, 1).transpose(0, 3, 1, 2).astype(np.float32) / 16.0

# Inflate images:
# - Upscale 8x8 -> 64x64 via nearest neighbor (repeat)
# - Convert 1 channel -> 3 channels by stacking
UPSCALE = 8                   # 8 * 8 = 64
CHANNELS = 3
X_big = np.repeat(np.repeat(X_images, UPSCALE, axis=2), UPSCALE, axis=3)  # (N,1,64,64)
X_big = np.repeat(X_big, CHANNELS, axis=1).copy()                         # (N,3,64,64)

# Light noise/jitter to avoid identical repeats (adds compute)
def augment(x):
    noise = np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
    x = np.clip(x + noise, 0.0, 1.0)
    return x

# Replicate dataset many times to extend training
REPEAT_TIMES = 16  # Increase if you want longer runs; adjust if it gets too slow
X_rep = np.tile(X_big, (REPEAT_TIMES, 1, 1, 1))
y_rep = np.tile(y, REPEAT_TIMES)

# Shuffle once
perm = np.random.permutation(len(X_rep))
X_rep = X_rep[perm]
y_rep = y_rep[perm]

# One-hot
n_classes = 10
y_onehot = np.eye(n_classes, dtype=np.float32)[y_rep]

# Augment (CPU-side) once for extra compute; you can move to per-batch if you prefer
X_rep = augment(X_rep)

# Train/val split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_rep, y_onehot, test_size=0.2, random_state=42, stratify=y_rep
)

print(f"Train: {X_train.shape}, Test: {X_test.shape} (N, C, H, W) with C={CHANNELS}, H=W={8*UPSCALE}")

# ====== Heavier model ======
# 64x64 -> (after 3 pools) -> 8x8; larger widths and big FCs
def create_heavy_model():
    return Sequential(
        # Block 1: keep big feature maps to burn memory/compute
        Conv2D(in_channels=CHANNELS, out_channels=64, kernel_size=(7,7), padding="same", activation="relu"),
        Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),    # 64 -> 32

        # Block 2
        Conv2D(in_channels=64, out_channels=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(in_channels=128, out_channels=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),    # 32 -> 16

        # Block 3
        Conv2D(in_channels=128, out_channels=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(in_channels=256, out_channels=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),    # 16 -> 8

        Flatten(),                                      # 256 * 8 * 8 = 16384
        Linear(256*8*8, 2048), ReLU(),
        Linear(2048, 1024), ReLU(),
        Linear(1024, n_classes),
        Softmax(axis=1),
    )

model = create_heavy_model()
print(model)
total_params = sum(p.data.size for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ====== Dynamic batch-size finder with OOM backoff ======
def find_max_batch(initial_bs=192, floor=8):
    """Try forward pass and shrink BS on OutOfMemory until it works."""
    bs = initial_bs
    # Choose dtype consistent with AMP toggle
    dtype = np.float16 if USE_MIXED_PRECISION and ng.DEVICE == "cuda" else np.float32
    test_x = X_train[:bs].astype(dtype, copy=False)
    test_y = y_train[:bs].astype(np.float32, copy=False)
    while bs >= floor:
        try:
            gpu_sync()
            log_vram(f"before trial BS={bs}")
            out = model(Tensor(test_x, requires_grad=False))
            loss = CategoricalCrossEntropy()(Tensor(test_y, False), out)
            # Small backward to test graph memory
            loss.backward()
            # Clear grads/graph
            for _, p in model.named_parameters():
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.fill(0)
            gpu_sync()
            log_vram(f"after  trial BS={bs}")
            return bs
        except Exception as e:
            msg = str(e)
            oomish = ("OutOfMemory" in msg) or ("out of memory" in msg) or ("CUDA_ERROR_OUT_OF_MEMORY" in msg)
            if ng.DEVICE == "cuda":
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            if oomish:
                bs = bs // 2
                test_x = X_train[:bs].astype(dtype, copy=False)
                test_y = y_train[:bs].astype(np.float32, copy=False)
                print(f"[OOM] Reducing batch size -> {bs}")
            else:
                raise
    return floor

# Start high and backoff; tweak initial_bs if you want to push harder
BATCH_SIZE = find_max_batch(initial_bs=192, floor=8)
print(f"Chosen batch size: {BATCH_SIZE}")

# ====== DataLoaders ======
train_loader = DataLoader(Dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(Dataset(X_test, y_test),   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ====== Optimizer / Loss / AMP ======
LR = 2e-4  # a bit higher than before; this is a profiling run
optimizer = Adam(model.named_parameters(), lr=LR)
loss_fn = CategoricalCrossEntropy()
scaler = GradScaler() if (USE_MIXED_PRECISION and GradScaler is not None) else None

# ====== Warm-up (important for profiling) ======
WARMUP_STEPS = 20
print(f"Warm-up: {WARMUP_STEPS} steps")
model.train()
for _ in range(WARMUP_STEPS):
    bx = X_train[:BATCH_SIZE]
    by = y_train[:BATCH_SIZE]
    with (autocast(enabled=True) if scaler else contextlib.nullcontext()):
        pred = model(bx)
        loss = loss_fn(by, pred)
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()
    optimizer.zero_grad()
gpu_sync()
log_vram("post warm-up")

# ====== Training loop ======
EPOCHS = 12   # sufficient to keep the GPU busy; raise if you want a longer run
print("="*60)
print("Start training")

def evaluate(model, Xn, Yn):
    model.eval()
    with (autocast(enabled=False) if scaler else contextlib.nullcontext()):
        preds = model(Tensor(Xn, requires_grad=False))
        loss  = loss_fn(Tensor(Yn, False), preds).data.item()
        # accuracy via class indices
        acc = float(ng_accuracy_score(Yn.argmax(1), preds.data.argmax(1)))
    return loss, acc

t0 = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    ep_losses, ep_accs = [], []
    t_ep = time.time()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        try:
            if scaler:
                with autocast(enabled=True):
                    pred = model(batch_x)
                    loss = loss_fn(batch_y, pred)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(batch_x)
                loss = loss_fn(batch_y, pred)
                loss.backward()
                optimizer.step()
            # simple accuracy
            acc = float(ng_accuracy_score(batch_y.argmax(1), pred.data.argmax(1)))
            ep_losses.append(loss.data.item()); ep_accs.append(acc)
        except Exception as e:
            msg = str(e)
            oomish = ("OutOfMemory" in msg) or ("out of memory" in msg) or ("CUDA_ERROR_OUT_OF_MEMORY" in msg)
            if oomish and ng.DEVICE == "cuda":
                print("[OOM] Skipping batch (consider lowering UPSCALE/REPEAT or disabling AMP).")
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                continue
            else:
                raise
    # eval
    gpu_sync()
    train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
    train_acc  = float(np.mean(ep_accs))  if ep_accs  else float("nan")
    test_loss, test_acc = evaluate(model, X_test, y_test)
    dt = time.time() - t_ep
    log_vram(f"epoch {epoch:02d} end")
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"test_loss {test_loss:.4f} acc {test_acc:.4f} | "
          f"time {dt:.2f}s")

T = time.time() - t0
print("="*60)
print(f"Done in {T:.2f}s")

# Optional quick plot so you still see something if running interactively
# (Scalene ignores matplotlib cost mostly; it's fine to keep.)
