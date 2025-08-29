# NeuroGrad GPU Memory & CUDA OOM Analysis

This document captures the root causes behind high VRAM usage and CUDA out‑of‑memory (OOM) issues observed in NeuroGrad. It synthesizes findings from the autograd core, convolution path, AMP/autocast, and the data pipeline.

## Executive Summary

- Dataset tensors are instantiated on the GPU by default, permanently occupying VRAM even outside the training step.
- Each Conv/Pool layer persists a `SlidingWindowView` Function object that caches an input‑sized gradient buffer on the GPU across iterations.
- Per‑op autocast (FP16↔FP32) repeatedly casts full feature maps within a single forward, multiplying peak memory.
- The `conv2d` implementation relies on a stride‑tricks sliding window + `tensordot`, which creates large intermediates and can allocate sizable workspaces (especially with cuTENSOR).
- CuPy’s memory pool retains freed blocks; apparent VRAM does not drop between steps, amplifying perceived usage spikes.

Together, these drive large, sustained VRAM baselines and high peak memory, triggering OOMs in conv‑heavy models with larger batches/resolutions.

## Primary Bottlenecks (With Evidence)

### 1) Dataset Instantiated on GPU (VRAM pinned baseline)

- Code constructs dataset tensors using global `xp` (CuPy on CUDA):
  - `neurograd/utils/data.py:15`
  - `neurograd/utils/data.py:16`
- With CUDA available, `xp` is CuPy (see device detection):
  - `neurograd/__init__.py:3`
  - `neurograd/__init__.py:26` (imports `cupy as xp` on CUDA)
- Result: Large `X/y` arrays move to VRAM at dataset creation time, independent of batch size and training loop.

### 2) Persistent sliding‑window grad buffers per layer

- `SlidingWindowView` caches an input‑sized gradient buffer and a view:
  - Allocations/caching: `neurograd/functions/tensor_ops.py:205`–`neurograd/functions/tensor_ops.py:209`
  - Reuse and zeroing: `neurograd/functions/tensor_ops.py:212`
- Conv/Pool layers store `SlidingWindowView` as a long‑lived module attribute and pass it into ops:
  - Conv2D ctor: `neurograd/nn/layers/conv.py:64`
  - Forward uses cached slider: `neurograd/nn/layers/conv.py:88`–`neurograd/nn/layers/conv.py:89`
  - MaxPool2D ctor/forward: `neurograd/nn/layers/conv.py:117`, `neurograd/nn/layers/conv.py:125`
  - AveragePool2D ctor/forward: `neurograd/nn/layers/conv.py:143`, `neurograd/nn/layers/conv.py:151`
- Result: For each such layer, an input‑sized GPU buffer persists across iterations, increasing baseline VRAM linearly with the number of layers and feature map sizes.

### 3) Per‑op autocast toggling causes repeated full‑tensor casts

- Autocast is consulted for every `Function.__call__` and may cast inputs per op:
  - `neurograd/functions/base.py:25`–`neurograd/functions/base.py:30`
- FP32‑forced ops include reductions, std, loss, and risky math; FP16‑safe includes conv, linalg, elementwise:
  - `_FP32_OPS`: `neurograd/amp/utils.py:13`
  - `_FP16_SAFE_OPS`: `neurograd/amp/utils.py:29`
- BN computes stats in FP32 and uses `.astype(xp.float32)` on arrays (additional copies):
  - `neurograd/nn/layers/batchnorm.py` (1D and 2D) forward paths
- Result: On a typical forward, feature maps may be converted FP16→FP32 (reductions/BN/loss) and then FP32→FP16 (activations/next conv), producing multiple full‑size transient copies per layer boundary and raising peak VRAM.

### 4) Convolution path: sliding window + tensordot = large intermediates

- `conv2d` shapes the input to `(N, C, out_H, out_W, F_H, F_W)` via sliding window and contracts with filters using `tensordot`:
  - `neurograd/functions/conv.py:226`–`neurograd/functions/conv.py:233`
- Depthwise path likewise builds large slide tensors before reduction.
- With cuTENSOR enabled, `tensordot` can select kernels requiring hefty temporary workspaces.
- Result: Even though the sliding window is a view, the contraction and backward create large temporaries; combined with (2) and (3), peak usage is high.

## Secondary Contributors

- cuTENSOR accelerator and CuPy memory pool
  - Accelerator: `neurograd/__init__.py:25` sets `CUPY_ACCELERATORS="cutensor"` (may increase workspace usage for contractions).
  - CuPy memory pool retains freed blocks, so VRAM appears to only grow during a run.
- Dropout/activations/losses store small per‑op intermediates (normal overhead), but additive at scale.

## Autograd Graph Retention

- `Tensor.backward` correctly frees intermediate grads and severs `grad_fn` links for non‑leaf nodes when `retain_graph=False`.
  - Clearing: see `neurograd/tensor.py` in `backward` loop.
- However, long‑lived module attributes (e.g., `self.slider`) survive across steps and manage their own GPU buffers as described in (2).

## Why OOM Happens in Practice (Typical Step Timeline)

1) Conv forward in FP16 → large activations.
2) BatchNorm/reductions in FP32 → FP16 activations recast to FP32 (full copy), stats computed; output often back to FP16 later.
3) Activation in FP16 → another cast boundary.
4) Next Conv repeats 1–3.
5) Backward creates counterpart temporaries; `SlidingWindowView` layers keep input‑sized grad buffers persistently.
6) CuPy pool + cuTENSOR workspace keep allocations around, so peak usage remains high over steps.

## Quick Checks You Can Run

- Confirm dataset residency: constructing `Dataset(X, y)` with large arrays should spike VRAM immediately.
- Track pool usage (interactive):
  - `import cupy as cp; cp.get_default_memory_pool().used_bytes()`
- Count cast boundaries by logging op names in `Function.__call__` or by instrumenting AMP utils.

## Recommendations (No Code Changes Applied Yet)

1) Keep dataset on CPU; only move batches to GPU inside the training step.
   - Avoid wrapping the entire `X/y` in `Tensor` backed by CuPy arrays at dataset construction.

2) Make `SlidingWindowView` stateless across steps (no persistent `_grad_buffer` on the layer), or ensure buffers are released after backward.
   - Store only configuration on the layer; construct the Function per forward call.

3) Reduce FP16↔FP32 thrashing.
   - Keep reductions/BN/loss in FP32, but avoid casting entire feature maps back to FP16 multiple times within the same block.
   - Consider module‑level autocast scopes to coalesce casts.

4) Evaluate cuTENSOR/workspace impact and memory pool behavior.
   - For debugging, set `CUPY_ACCELERATORS=""` to gauge workspace contribution.
   - Optionally trim the memory pool between epochs if you need headroom for large batches.

5) Batch/resolution tuning.
   - Until structural changes land, reduce batch size or input resolution to stay within VRAM.

## Relevant Code References

- Dataset creation on GPU:
  - `neurograd/utils/data.py:15`
  - `neurograd/utils/data.py:16`
- Device selection / CuPy import:
  - `neurograd/__init__.py:3`
  - `neurograd/__init__.py:25`–`neurograd/__init__.py:26`
- Autocast at Function boundary:
  - `neurograd/functions/base.py:25`–`neurograd/functions/base.py:30`
  - AMP op sets: `neurograd/amp/utils.py:13`, `neurograd/amp/utils.py:29`
- Sliding window grad buffers:
  - `neurograd/functions/tensor_ops.py:205`–`neurograd/functions/tensor_ops.py:212`
  - Layer ownership: `neurograd/nn/layers/conv.py:64`, `:88`–`:89`, `:117`, `:125`, `:143`, `:151`
- Convolution contraction path:
  - `neurograd/functions/conv.py:226`–`neurograd/functions/conv.py:233`
- BatchNorm FP32 stats:
  - `neurograd/nn/layers/batchnorm.py` (both `BatchNorm` and `BatchNorm2D` forward)

---

No code was modified to produce this analysis. When you’re ready, we can apply targeted changes with minimal surface area to address the top issues.

