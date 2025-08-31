import threading


_TLS = threading.local()
if not hasattr(_TLS, "enabled"):
    _TLS.enabled = True


class no_grad:
    def __init__(self, enabled: bool = True):
        self._set = bool(enabled)
        self._prev = None

    def __enter__(self):
        self._prev = getattr(_TLS, "enabled", True)
        if self._set:
            _TLS.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        _TLS.enabled = self._prev
        return False


def is_grad_enabled() -> bool:
    return bool(getattr(_TLS, "enabled", True))


def set_grad_enabled(mode: bool) -> None:
    _TLS.enabled = bool(mode)

