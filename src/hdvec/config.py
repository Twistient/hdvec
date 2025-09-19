"""Global configuration for hdvec.

This module centralizes runtime-configurable options and provides a lightweight
singleton-style accessor. The surface is intentionally small and can expand as
implementations mature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

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
    moduli: Optional[list[int]] = None
    conv_backend: Literal["fft"] = "fft"


# Module-global config instance; lightweight and mutable
_GLOBAL_CONFIG = Config()


def get_config() -> Config:
    """Return the current global Config instance."""
    return _GLOBAL_CONFIG

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import importlib.util
import numpy as np


@dataclass
class Config:
    """Global settings for hdvec.

    Fields:
        D: Dimensionality (default 1024)
        binding: 'hadamard' | 'cc' | 'lcc'
        backend: 'numpy' | 'torch' (torch optional)
        dtype: numpy dtype (default complex64)
    """
    D: int = 1024
    binding: Literal['hadamard', 'cc', 'lcc'] = 'hadamard'
    backend: Literal['numpy', 'torch'] = 'numpy'
    dtype: np.dtype = np.complex64


_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """Return the process-global configuration singleton.

    Auto-detect torch availability but do not switch backend by default.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
        # Optionally, set backend if torch is the only available numeric backend.
        if importlib.util.find_spec("torch") is not None:
            # Leave as 'numpy' by default; user can opt-in via Config if desired.
            pass
    return _CONFIG
