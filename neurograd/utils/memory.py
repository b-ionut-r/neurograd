"""
Lightweight, opt-in GPU memory monitor.

Usage:
    from neurograd.utils.memory import MemoryMonitor
    with MemoryMonitor():
        # run code; each NeuroGrad op logs a one-line memory snapshot

Default off: When not inside the context, overhead is near-zero
(a single boolean check per op).

Notes:
- On CUDA (CuPy), prints driver used VRAM, CuPy pool used/total, and optional FFT
  plan cache info if available.
- On CPU (NumPy), logs only the op tag without memory figures.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterable, Optional, Dict, Any


_TLS = threading.local()
_TLS.enabled = False
_TLS.print_fn = print  # type: ignore
_TLS.prefix = "[OP]"
_TLS.include_driver = True
_TLS.include_pool = True
_TLS.include_fft = False


def _get_backend() -> str:
    try:
        import neurograd as ng  # local import to avoid cycles

        return ng.DEVICE
    except Exception:
        return "cpu"


def _fmt_gb(x: float) -> str:
    return f"{x/1e9:.2f} GB"


def _gpu_stats() -> Optional[str]:
    if _get_backend() != "cuda":
        return None
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None

    parts = []
    try:
        if getattr(_TLS, "include_driver", True):
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            parts.append(f"drv used/total={_fmt_gb(used)}/{_fmt_gb(total)}")
    except Exception:
        pass

    try:
        if getattr(_TLS, "include_pool", True):
            mp = cp.get_default_memory_pool()
            used_b = mp.used_bytes()
            tot_b = mp.total_bytes()
            parts.append(f"pool used/total={_fmt_gb(used_b)}/{_fmt_gb(tot_b)}")
    except Exception:
        pass

    if getattr(_TLS, "include_fft", False):
        try:
            from cupyx.scipy import fft as cufft  # type: ignore

            pc = cufft.get_plan_cache()
            parts.append(f"fft plans={pc.get_size()} mem={_fmt_gb(pc.get_memsize())}")
        except Exception:
            pass

    return " | ".join(parts) if parts else None


def is_enabled() -> bool:
    return bool(getattr(_TLS, "enabled", False))


def maybe_log_op_memory(op_name: str, inputs: Iterable, output) -> None:
    """
    Fast no-op when disabled. Called by the Function machinery after each op.
    """
    if not is_enabled():
        return

    try:
        pf: Callable[[str], None] = getattr(_TLS, "print_fn", print)  # type: ignore
        prefix: str = getattr(_TLS, "prefix", "[OP]")  # type: ignore

        in_shapes = []
        for t in inputs or []:
            try:
                in_shapes.append(tuple(getattr(t, "shape", ())))
            except Exception:
                in_shapes.append(())
        out_shape = None
        try:
            out_shape = tuple(getattr(output, "shape", ()))
        except Exception:
            out_shape = ()

        gpu = _gpu_stats()
        msg = f"{prefix} {op_name}: in={in_shapes} -> out={out_shape}"
        if gpu:
            msg += f" | {gpu}"
        pf(msg)
    except Exception:
        # Never let diagnostics break the main path
        return


class MemoryMonitor:
    """Context manager to enable per-op memory logging."""

    def __init__(
        self,
        *,
        print_fn: Optional[Callable[[str], None]] = None,
        prefix: str = "[OP]",
        include_driver: bool = True,
        include_pool: bool = True,
        include_fft: bool = False,
    ) -> None:
        self._prev = {}
        self.print_fn = print_fn or print
        self.prefix = prefix
        self.include_driver = include_driver
        self.include_pool = include_pool
        self.include_fft = include_fft

    def __enter__(self):
        self._prev = {
            "enabled": getattr(_TLS, "enabled", False),
            "print_fn": getattr(_TLS, "print_fn", print),
            "prefix": getattr(_TLS, "prefix", "[OP]"),
            "include_driver": getattr(_TLS, "include_driver", True),
            "include_pool": getattr(_TLS, "include_pool", True),
            "include_fft": getattr(_TLS, "include_fft", False),
        }
        _TLS.enabled = True
        _TLS.print_fn = self.print_fn  # type: ignore
        _TLS.prefix = self.prefix
        _TLS.include_driver = self.include_driver
        _TLS.include_pool = self.include_pool
        _TLS.include_fft = self.include_fft
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self._prev.items():
            setattr(_TLS, k, v)
        return False


def log_point(tag: str) -> None:
    """Manual memory log marker."""
    if not is_enabled():
        return
    gpu = _gpu_stats()
    pf: Callable[[str], None] = getattr(_TLS, "print_fn", print)  # type: ignore
    prefix: str = getattr(_TLS, "prefix", "[OP]")  # type: ignore
    msg = f"{prefix} {tag}"
    if gpu:
        msg += f" | {gpu}"
    pf(msg)


# -------------------------- Optional Pool Guard --------------------------- #

_PG: Dict[str, Any] = {
    "enabled": False,
    "interval_ops": 0,           # flush check every N ops (0 disables)
    "min_free_bytes": None,      # flush if driver free < threshold
    "flush_slack_bytes": None,   # flush if (pool_total - pool_used) > slack
    "cap_bytes": None,           # set pool cap (set_limit)
    "op_counter": 0,
    "print_fn": None,
}


def enable_pool_guard(
    *,
    interval_ops: int = 0,
    min_free_bytes: Optional[int] = None,
    flush_slack_bytes: Optional[int] = None,
    cap_bytes: Optional[int] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Enable pool guard. All params optional; when none are set, does nothing.

    - interval_ops: check every N ops (0 disables)
    - min_free_bytes: if driver free < threshold, flush
    - flush_slack_bytes: if pool_total - pool_used > threshold, flush
    - cap_bytes: set CuPy pool cap
    - print_fn: optional logger
    """
    _PG.update(
        enabled=True,
        interval_ops=max(0, int(interval_ops)),
        min_free_bytes=min_free_bytes,
        flush_slack_bytes=flush_slack_bytes,
        cap_bytes=cap_bytes,
        op_counter=0,
        print_fn=print_fn,
    )


def disable_pool_guard() -> None:
    _PG.update(enabled=False, op_counter=0)


def _pg_log(msg: str) -> None:
    fn = _PG.get("print_fn")
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass


def maybe_flush_pool(op_name: str = "") -> None:
    """
    Very cheap when disabled. Called per-op to optionally cap/flush the CuPy pool.
    """
    if not _PG.get("enabled"):
        return
    interval = _PG.get("interval_ops", 0)
    _PG["op_counter"] = int(_PG.get("op_counter", 0)) + 1
    if interval <= 0 or (_PG["op_counter"] % interval) != 0:
        return

    # CPU backend: nothing to do
    if _get_backend() != "cuda":
        return

    try:
        import cupy as cp  # type: ignore
    except Exception:
        return

    # Cap pool if requested
    try:
        cap = _PG.get("cap_bytes")
        if cap is not None:
            cp.get_default_memory_pool().set_limit(int(cap))
    except Exception:
        pass

    # Gather stats
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        mp = cp.get_default_memory_pool()
        used_b = mp.used_bytes()
        tot_b = mp.total_bytes()
        slack = max(0, tot_b - used_b)
    except Exception:
        return

    do_flush = False
    min_free = _PG.get("min_free_bytes")
    slack_thr = _PG.get("flush_slack_bytes")
    if isinstance(min_free, int) and free < min_free:
        do_flush = True
    if isinstance(slack_thr, int) and slack > slack_thr:
        do_flush = True

    if do_flush:
        # Use ng.flush() if available; else fallback to direct pool flush
        try:
            import neurograd as ng  # type: ignore

            if hasattr(ng, "flush"):
                ng.flush()
            else:
                raise AttributeError
        except Exception:
            try:
                mp.free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        _pg_log(
            f"[PoolGuard] flush at op='{op_name}' | drv_free={_fmt_gb(float(free))} | pool used/total={_fmt_gb(float(used_b))}/{_fmt_gb(float(tot_b))}"
        )
