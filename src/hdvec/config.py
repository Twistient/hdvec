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
