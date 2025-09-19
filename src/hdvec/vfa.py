"""Vector Function Architecture (VFA) operations.

Minimal encoders using FPE.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import bind, similarity
from .fpe import encode_fpe
from .utils import ensure_array
from .base import Vec


def encode_function(points: np.ndarray, alphas: np.ndarray, base: np.ndarray) -> Vec:
    """VFA function rep: y_f = sum_k alpha_k z(r_k).

    Args:
        points: (K,) samples r_k.
        alphas: (K,) coefficients.
        base: (D,) FPE base vector.
    Returns:
        (D,) complex vector representation.
    """
    if points.shape != alphas.shape:
        raise ValueError("points and alphas must have same shape")
    zs = [alphas[k] * encode_fpe(float(points[k]), base) for k in range(points.size)]
    return Vec(np.sum(zs, axis=0).astype(np.complex64))


def readout(y_f: np.ndarray, s: float, base: np.ndarray) -> float:
    """Readout: ⟨y_f, z(s)⟩ / D (real part)."""
    z_s = encode_fpe(s, base)
    return similarity(y_f, z_s)


def shift(y_f: np.ndarray, t: float, base: np.ndarray) -> Vec:
    """Shift: y_f ∘ z(t) via hadamard bind."""
    z_t = encode_fpe(t, base)
    return bind(y_f, z_t, op="hadamard")


def convolve(y_f: np.ndarray, y_g: np.ndarray) -> Vec:
    """Circular convolution of two representations (placeholder via FFT)."""
    a = ensure_array(y_f)
    b = ensure_array(y_g)
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return Vec(np.fft.ifft(fa * fb).astype(np.complex64))


@dataclass
class VFAEncoder:
    base: np.ndarray

    def encode(self, points: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        return encode_function(points, alphas, self.base)

    def readout(self, y_f: np.ndarray, s: float) -> float:  # type: ignore[override]
        return readout(y_f, s, self.base)

    def shift(self, y_f: np.ndarray, t: float) -> np.ndarray:  # type: ignore[override]
        return shift(y_f, t, self.base)
