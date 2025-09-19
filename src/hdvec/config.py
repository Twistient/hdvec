"""Global configuration for hdvec.

This module centralizes runtime-configurable options and provides a lightweight
singleton-style accessor. The surface is intentionally small and can expand as
implementations mature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


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
    """

    D: int = 1024
    backend: Literal["numpy", "torch"] = "numpy"
    dtype: np.dtype = np.complex64
    binding: Literal["hadamard", "cc", "lcc"] = "hadamard"
    dist: Literal["uniform", "cauchy"] = "uniform"
    m: int = 1
    moduli: list[int] | None = None
    conv_backend: Literal["fft"] = "fft"


# Module-global config instance; lightweight and mutable
_GLOBAL_CONFIG = Config()


def get_config() -> Config:
    """Return the current global Config instance."""
    return _GLOBAL_CONFIG
