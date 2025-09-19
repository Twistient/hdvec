"""Core VSA primitives (superposition, binding, similarity, permutation) and config.

Minimal NumPy implementations with optional Numba JIT.
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .base import BaseVector
from .utils import phase_normalize, hermitian_enforce, optional_njit, ensure_array
from .config import Config


@optional_njit(cache=True)
def _similarity_numba(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[no-redef]
    # Real part of normalized inner product
    denom = max(1, a.size)
    # Note: Numba supports complex operations; cast to float for return
    val = (a.conj() * b).sum().real / denom
    return float(val)


def bind(a: Union[np.ndarray, BaseVector], b: Union[np.ndarray, BaseVector], op: str = "hadamard") -> np.ndarray:
    """Vector binding operation, configurable per VSA type.

    Supports:
      - hadamard: componentwise multiplication (phasor add)
      - cc: circular convolution via FFT
      - lcc: placeholder (raises NotImplementedError)

    Note: This transitional signature accepts BaseVector but returns a NumPy array.
    """
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    if op == "hadamard":
        out = a_arr * b_arr
        return phase_normalize(out) if np.iscomplexobj(out) else out
    if op == "cc":
        fa = np.fft.fft(a_arr)
        fb = np.fft.fft(b_arr)
        out = np.fft.ifft(fa * fb)
        return out.astype(np.complex64) if np.iscomplexobj(a_arr) or np.iscomplexobj(b_arr) else out.real
    if op == "lcc":
        raise NotImplementedError("Localized circular convolution (lcc) is not implemented in stubs.")
    raise ValueError(f"Unknown binding op: {op}")


def bundle(a: Union[np.ndarray, BaseVector], b: Union[np.ndarray, BaseVector]) -> np.ndarray:
    """Superposition (bundling) of two vectors with renormalization for phasors.

    Note: Accepts BaseVector inputs, returns a NumPy array (transitional API).
    """
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    out = a_arr + b_arr
    return phase_normalize(out) if np.iscomplexobj(out) else out / 2.0


def similarity(a: Union[np.ndarray, BaseVector], b: Union[np.ndarray, BaseVector]) -> float:
    """Real part of normalized inner product ⟨a,b⟩/D.

    Returns a scalar in approximately [-1, 1] for unit-normalized phasors.

    Note: Accepts BaseVector inputs (transitional API).
    """
    a_arr = ensure_array(a).astype(np.complex64)
    b_arr = ensure_array(b).astype(np.complex64)
    # Use JIT when available
    return _similarity_numba(a_arr, b_arr)


def permute(v: Union[np.ndarray, BaseVector], shift: int) -> np.ndarray:
    """Permutation as circular shift (roll) along the last axis.

    Note: Accepts BaseVector input, returns NumPy array.
    """
    v_arr = ensure_array(v)
    return np.roll(v_arr, shift, axis=-1)
