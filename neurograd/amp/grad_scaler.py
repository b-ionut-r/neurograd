"""
Simple AMP GradScaler (no overflow checks)
"""

from typing import Any, Dict, Optional
from neurograd import xp


class GradScaler:
    def __init__(
        self,
        init_scale: float = 256.0,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        max_scale: float = 2**16,
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
        # Always perform the scaling multiply in FP32 and outside autocast
        # to prevent FP16 overflow when the scale grows large.
        if not self._enabled:
            return tensor
        from neurograd.tensor import Tensor
        try:
            from neurograd.amp.autocast import autocast as _autocast
        except Exception:
            # Fallback: no autocast available
            _autocast = None

        if not isinstance(tensor, Tensor):
            tensor = Tensor(
                xp.asarray(tensor),
                requires_grad=getattr(tensor, "requires_grad", False),
            )

        # Ensure the loss tensor is FP32 for numerically safe scaling
        if tensor.data.dtype != xp.float32:
            tensor = tensor.cast(xp.float32)

        # Use a FP32 scale value as an array to avoid dtype promotion to FP64
        scale_arr = xp.array(self._scale, dtype=xp.float32)

        if _autocast is not None:
            # Disable autocast just for this multiply
            with _autocast(enabled=False):
                return tensor * scale_arr
        else:
            return tensor * scale_arr

    def unscale_(self, optimizer) -> None:
        if not self._enabled or self._has_been_unscaled:
            return
        from neurograd.tensor import Tensor

        inv = 1.0 / self._scale if self._scale > 0 else 0.0
        self._found_inf = False
        for _, param in optimizer.params:
            grad = getattr(param, "grad", None)
            if grad is None or not getattr(param, "requires_grad", False):
                continue

            arr = grad.data if isinstance(grad, Tensor) else grad  # xp.ndarray

            # Fast finite check; bail early on first non-finite
            try:
                if not bool(xp.isfinite(arr).all()):
                    self._found_inf = True
            except Exception:
                pass

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

    def step(self, optimizer) -> bool:
        if not self._enabled:
            optimizer.step()
            return True
        if not self._has_been_unscaled:
            self.unscale_(optimizer)
        if self._found_inf:
            # Skip stepping on overflow; let update() reduce scale
            return False
        optimizer.step()
        return True

    def update(self, new_scale: Optional[float] = None) -> None:
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
        else:
            if self._found_inf:
                # Backoff on overflow
                self._scale = max(self._scale / self._growth_factor, 1.0)
                self._growth_tracker = 0
            else:
                self._growth_tracker += 1
                if self._growth_tracker >= self._growth_interval:
                    self._scale = min(self._scale * self._growth_factor, self._max_scale)
                    self._growth_tracker = 0

        # ready for next step
        self._has_been_unscaled = False
        self._found_inf = False

    # --- checkpointing ---
    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "growth_factor": self._growth_factor,
            "growth_interval": self._growth_interval,
            "max_scale": self._max_scale,
            "enabled": self._enabled,
            "found_inf": self._found_inf,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._scale = state_dict["scale"]
        self._growth_tracker = state_dict["growth_tracker"]
        self._growth_factor = state_dict["growth_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._max_scale = state_dict.get("max_scale", 2**20)
        self._enabled = state_dict["enabled"]
        self._has_been_unscaled = False
        self._found_inf = state_dict.get("found_inf", False)

    def __repr__(self) -> str:
        if not self._enabled:
            return "GradScaler(enabled=False)"
        return (
            f"GradScaler(scale={self._scale:.1f}, "
            f"growth_factor={self._growth_factor}, "
            f"growth_interval={self._growth_interval})"
        )
