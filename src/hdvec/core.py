"""Core VSA primitives (superposition, binding, similarity, permutation) and config.

Minimal NumPy implementations with optional Numba JIT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import phase_normalize, hermitian_enforce, optional_njit


@dataclass
class Config:
    """Global settings for hdvec operations.

    Attributes:
        D: Dimensionality of vectors.
        backend: "numpy" or "torch" (torch not used in stubs).
        dtype: Numpy dtype (complex64 by default for phasors).
        conv_backend: Backend for circular convolution ("fft").
    """
    D: int = 1024
    backend: Literal["numpy", "torch"] = "numpy"
    dtype: np.dtype = np.complex64
    conv_backend: Literal["fft"] = "fft"


@optional_njit(cache=True)
def _similarity_numba(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[no-redef]
    # Real part of normalized inner product
    denom = max(1, a.size)
    # Note: Numba supports complex operations; cast to float for return
    val = (a.conj() * b).sum().real / denom
    return float(val)


def bind(a: np.ndarray, b: np.ndarray, op: str = "hadamard") -> np.ndarray:
    """Vector binding operation, configurable per VSA type.

    Supports:
      - hadamard: componentwise multiplication (phasor add)
      - cc: circular convolution via FFT
      - lcc: placeholder (raises NotImplementedError)
    """
    if op == "hadamard":
        out = a * b
        return phase_normalize(out) if np.iscomplexobj(out) else out
    if op == "cc":
        fa = np.fft.fft(a)
        fb = np.fft.fft(b)
        out = np.fft.ifft(fa * fb)
        return out.astype(np.complex64) if np.iscomplexobj(a) or np.iscomplexobj(b) else out.real
    if op == "lcc":
        raise NotImplementedError("Localized circular convolution (lcc) is not implemented in stubs.")
    raise ValueError(f"Unknown binding op: {op}")


def bundle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Superposition (bundling) of two vectors with renormalization for phasors."""
    out = a + b
    return phase_normalize(out) if np.iscomplexobj(out) else out / 2.0


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Real part of normalized inner product ⟨a,b⟩/D.

    Returns a scalar in approximately [-1, 1] for unit-normalized phasors.
    """
    # Use JIT when available
    return _similarity_numba(a.astype(np.complex64), b.astype(np.complex64))


def permute(v: np.ndarray, shift: int) -> np.ndarray:
    """Permutation as circular shift (roll) along the last axis."""
    return np.roll(v, shift, axis=-1)
