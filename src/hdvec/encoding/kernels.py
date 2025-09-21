"""Kernel diagnostics for FPE encodings."""

from __future__ import annotations

import numpy as np

from ..utils import ensure_array, phase_normalize

__all__ = ["estimate_kernel"]


def estimate_kernel(base: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate empirical kernel values for offsets in ``samples``.

    Returns a tuple (offsets, similarities).
    """
    base_arr = ensure_array(base)
    base_arr = phase_normalize(base_arr)
    sims = []
    for delta in samples:
        enc = np.power(base_arr, delta)
        sims.append((np.conj(base_arr) * enc).mean().real)
    return samples, np.asarray(sims, dtype=np.float64)
