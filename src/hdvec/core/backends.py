"""Backend registry for hdvec operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

__all__ = [
    "Backend",
    "register_backend",
    "get_backend",
    "available_backends",
    "set_backend",
]


@dataclass
class Backend:
    """Descriptor for an execution backend."""

    name: str
    supports_complex64: bool = True
    supports_fft: bool = True
    array_module: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        if self.array_module is None and self.name == "numpy":
            import numpy as np

            self.array_module = np


_BACKENDS: Dict[str, Backend] = {}
_CURRENT: str = "numpy"


def register_backend(backend: Backend, *, overwrite: bool = False) -> None:
    if not overwrite and backend.name in _BACKENDS:
        raise ValueError(f"Backend '{backend.name}' already registered")
    _BACKENDS[backend.name] = backend


def get_backend(name: str | None = None) -> Backend:
    key = name or _CURRENT
    if key not in _BACKENDS:
        raise KeyError(f"Backend '{key}' is not registered")
    return _BACKENDS[key]


def available_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def set_backend(name: str) -> None:
    if name not in _BACKENDS:
        raise KeyError(f"Backend '{name}' is not registered")
    global _CURRENT
    _CURRENT = name


# Register the default NumPy backend at import time
register_backend(Backend(name="numpy"))
