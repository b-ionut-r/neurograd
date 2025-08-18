"""
AMP GradScaler with overflow detection and safe FP32 scaling

Key changes:
- Perform `scale(loss)` multiplication with autocast disabled to keep it in FP32.
- Detect inf/NaN in grads after unscale and skip optimizer.step on overflow.
- Dynamically grow scale only when safe; reduce on overflow.
"""

from typing import Any, Dict, Optional
from neurograd import xp


class GradScaler:
    def __init__(
        self,
        init_scale: float = 256.0,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        backoff_factor: float = 0.5,
        min_scale: float = 1.0,
        max_scale: float = 2**20,
        enabled: bool = True,
    ):
        self._enabled = bool(enabled)
        if not self._enabled:
            return
        if init_scale <= 0:
            raise ValueError("init_scale must be > 0")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        if growth_interval < 1:
            raise ValueError("growth_interval must be >= 1")
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._growth_interval = int(growth_interval)
        self._backoff_factor = float(backoff_factor)
        if not (0.0 < self._backoff_factor < 1.0):
            raise ValueError("backoff_factor must be in (0, 1)")
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)

        self._growth_tracker = 0
        self._has_been_unscaled = False
        self._found_inf = False

    # --- info ---
    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale if self._enabled else 1.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "current_scale": self._scale if self._enabled else 1.0,
            "growth_tracker": self._growth_tracker if self._enabled else 0,
        }

    # --- core API ---
    def scale(self, tensor) -> "Tensor":
        if not self._enabled:
            return tensor
        from neurograd.tensor import Tensor
        if not isinstance(tensor, Tensor):
            tensor = Tensor(
                xp.asarray(tensor),
                requires_grad=getattr(tensor, "requires_grad", False),
            )
        # Multiply in FP32 with autocast disabled to avoid FP16 overflow
        try:
            from .autocast import autocast as _ac
            with _ac(enabled=False):
                # Ensure the scaled value lives in FP32 space
                if tensor.data.dtype != xp.float32:
                    tensor = tensor.cast(xp.float32)
                return tensor * float(self._scale)
        except Exception:
            # Fallback: do a safe FP32 multiply without relying on autocast
            if tensor.data.dtype != xp.float32:
                tensor = tensor.cast(xp.float32)
            return tensor * float(self._scale)

    def unscale_(self, optimizer) -> None:
        if not self._enabled or self._has_been_unscaled:
            return
        from neurograd.tensor import Tensor

        inv = 1.0 / self._scale if self._scale > 0 else 0.0
        # Unscale grads in-place when possible
        for _, param in optimizer.params:
            grad = getattr(param, "grad", None)
            if grad is None or not getattr(param, "requires_grad", False):
                continue

            arr = grad.data if isinstance(grad, Tensor) else grad  # xp.ndarray

            # in-place when possible; fallback to out-of-place then write back
            try:
                arr *= arr.dtype.type(inv)
            except Exception:
                arr = arr * arr.dtype.type(inv)
                if isinstance(grad, Tensor):
                    grad.data = arr
                else:
                    param.grad = arr

        self._has_been_unscaled = True

    def _check_for_inf_nan(self, optimizer) -> bool:
        """Return True if any gradient contains inf or NaN."""
        for _, param in optimizer.params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            arr = grad.data if hasattr(grad, "data") else grad
            try:
                # xp.isfinite supports cupy/numpy
                if not xp.isfinite(arr).all():
                    return True
            except Exception:
                # Conservative: if check fails, assume overflow
                return True
        return False

    def step(self, optimizer) -> bool:
        if not self._enabled:
            optimizer.step()
            return True
        if not self._has_been_unscaled:
            self.unscale_(optimizer)

        # Detect overflow and skip step if found
        self._found_inf = self._check_for_inf_nan(optimizer)
        if self._found_inf:
            # Skip optimizer step on overflow
            return False

        optimizer.step()
        return True

    def update(self, new_scale: Optional[float] = None) -> None:
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
            self._found_inf = False
        else:
            if self._found_inf:
                # Backoff the scale on overflow and reset tracker
                self._scale = max(self._scale * self._backoff_factor, self._min_scale)
                self._growth_tracker = 0
                self._found_inf = False
            else:
                # Grow scale only if no overflow for growth_interval steps
                self._growth_tracker += 1
                if self._growth_tracker >= self._growth_interval:
                    self._scale = min(self._scale * self._growth_factor, self._max_scale)
                    self._growth_tracker = 0

        # ready for next step
        self._has_been_unscaled = False

    # --- checkpointing ---
    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "growth_factor": self._growth_factor,
            "growth_interval": self._growth_interval,
            "backoff_factor": self._backoff_factor,
            "min_scale": self._min_scale,
            "max_scale": self._max_scale,
            "enabled": self._enabled,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._scale = state_dict["scale"]
        self._growth_tracker = state_dict["growth_tracker"]
        self._growth_factor = state_dict["growth_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._backoff_factor = state_dict.get("backoff_factor", 0.5)
        self._min_scale = state_dict.get("min_scale", 1.0)
        self._max_scale = state_dict.get("max_scale", 2**20)
        self._enabled = state_dict["enabled"]
        self._has_been_unscaled = False
        self._found_inf = False

    def __repr__(self) -> str:
        if not self._enabled:
            return "GradScaler(enabled=False)"
        return (
            f"GradScaler(scale={self._scale:.1f}, "
            f"growth_factor={self._growth_factor}, "
            f"growth_interval={self._growth_interval})"
        )
