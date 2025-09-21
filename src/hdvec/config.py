"""Global configuration for hdvec.

This module centralizes runtime-configurable options and provides a lightweight
singleton-style accessor. The surface is intentionally small and can expand as
implementations mature.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .errors import ConfigurationError


@dataclass
class Config:
    """Global settings for hdvec operations.

    Attributes:
        D: Dimensionality of vectors.
        backend: Backend selection ("numpy" or "torch").
        dtype: Default dtype for phasor vectors.
        binding: Default binding op ("hadamard" | "cc" | "lcc").
        dist: Default FPE phase distribution ("uniform" | "cauchy").
        m: Matrix size for GHRR slices (m x m).
        moduli: Optional moduli list for residue encoders.
        conv_backend: Backend for convolution-style binding (currently "fft").
        lcc_blocks: Number of blocks for localized circular convolution; ``None``
            disables LCC unless callers override it explicitly.
    """

    D: int = 1024
    backend: Literal["numpy", "torch"] = "numpy"
    dtype: np.dtype[np.complexfloating[Any, Any]] = np.dtype(np.complex64)
    binding: Literal["hadamard", "cc", "lcc"] = "hadamard"
    dist: Literal["uniform", "cauchy"] = "uniform"
    m: int = 1
    moduli: list[int] | None = None
    conv_backend: Literal["fft"] = "fft"
    lcc_blocks: int | None = None


# Module-global config instance; lightweight and mutable
_GLOBAL_CONFIG = Config()


def get_config() -> Config:
    """Return the current global Config instance."""
    return _GLOBAL_CONFIG


class override:
    """Context manager to temporarily override configuration values."""

    def __init__(self, **kwargs: Any) -> None:
        self._updates = kwargs
        self._original = replace(_GLOBAL_CONFIG)

    def __enter__(self) -> Config:
        from .core.backends import set_backend

        for key, value in self._updates.items():
            if not hasattr(_GLOBAL_CONFIG, key):
                raise ConfigurationError(f"Unknown config field: {key}")
            setattr(_GLOBAL_CONFIG, key, value)
        if "backend" in self._updates:
            set_backend(self._updates["backend"])
        return _GLOBAL_CONFIG

    def __exit__(self, exc_type, exc, tb) -> None:
        from .core.backends import set_backend

        global _GLOBAL_CONFIG
        _GLOBAL_CONFIG = self._original
        set_backend(_GLOBAL_CONFIG.backend)


def set_backend_name(name: str) -> None:
    from .core.backends import available_backends, set_backend

    if name not in available_backends():
        raise ConfigurationError(f"Backend '{name}' is not registered")
    _GLOBAL_CONFIG.backend = name
    set_backend(name)
