"""Vector Function Architecture (VFA) operations.

Minimal encoders using FPE and function-level helpers for readout/shift/convolution.
Includes a lightweight grid/field encoder built from FPE and core algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseVector, Vec
from .core import Codebook, bind, inv, similarity
from .fpe import encode_fpe, encode_fpe_vec
from .utils import ensure_array


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


def readout(y_f: np.ndarray | BaseVector, s: float, base: np.ndarray) -> float:
    """Readout: ⟨y_f, z(s)⟩ / D (real part)."""
    z_s = encode_fpe(s, base)
    return similarity(y_f, z_s)


def shift(y_f: np.ndarray | BaseVector, t: float, base: np.ndarray) -> Vec:
    """Shift: y_f ∘ z(t) via hadamard bind."""
    z_t = encode_fpe(t, base)
    return bind(y_f, z_t, op="hadamard")


def convolve(y_f: np.ndarray | BaseVector, y_g: np.ndarray | BaseVector) -> Vec:
    """Circular convolution of two representations (placeholder via FFT)."""
    a = ensure_array(y_f)
    b = ensure_array(y_g)
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return Vec(np.fft.ifft(fa * fb).astype(np.complex64))


def encode_grid(
    values: np.ndarray,
    pos_bases: list[np.ndarray],
    value_codebook: np.ndarray | None = None,
) -> Vec:
    """Encode a 2D (or ND) array by binding position codes with value encodings and bundling.

    If ``value_codebook`` is provided (K, D), values is assumed to be integer indices into
    the codebook. Otherwise, ``values`` is expected to contain encoded vectors per cell with
    final dimension D.
    """
    arr = np.asarray(values)
    if value_codebook is not None:
        if arr.ndim < 2:
            raise ValueError("values must be an integer grid when using a codebook")
        h_, w_ = int(arr.shape[0]), int(arr.shape[1])
        out = np.zeros(value_codebook.shape[1], dtype=np.complex64)
        for i in range(h_):
            for j in range(w_):
                pos = encode_fpe_vec(np.array([float(i), float(j)], dtype=float), pos_bases)
                out = out + pos * value_codebook[int(arr[i, j])]
        return Vec(out.astype(np.complex64))
    # values contains encoded vectors: shape (..., D)
    if arr.ndim < 3:
        raise ValueError("values must have shape (..., D) when no codebook is provided")
    d_ = int(arr.shape[-1])
    coords = np.stack(
        np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij"), axis=-1
    )
    out = np.zeros(d_, dtype=np.complex64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            pos = encode_fpe_vec(coords[i, j].astype(float), pos_bases)
            out = out + pos * arr[i, j]
    return Vec(out.astype(np.complex64))


def translate_grid(
    scene: np.ndarray | BaseVector, dx: float, dy: float, pos_bases: list[np.ndarray]
) -> Vec:
    """Translate a scene HV by (dx, dy) using FPE shift: bind with z(dx,dy)."""
    shift = encode_fpe_vec(np.array([dx, dy], dtype=float), pos_bases)
    return bind(scene, shift)


def read_cell(
    scene: np.ndarray | BaseVector,
    i: int,
    j: int,
    pos_bases: list[np.ndarray],
    value_codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Read out the value HV at cell (i,j) by unbinding with the position code.

    If a value codebook is given, returns the (index, score) of the nearest atom via cleanup.
    Otherwise, returns the raw HV for that location (no cleanup).
    """
    pos = encode_fpe_vec(np.array([float(i), float(j)], dtype=float), pos_bases)
    probe = bind(scene, inv(pos))
    if value_codebook is not None:
        idx, score = Codebook(value_codebook).nearest(ensure_array(probe))
        return np.array([idx, score])
    return ensure_array(probe)


@dataclass
class VFAEncoder:
    base: np.ndarray

    def encode(self, points: np.ndarray, alphas: np.ndarray) -> Vec:
        return encode_function(points, alphas, self.base)

    def readout(self, y_f: np.ndarray | BaseVector, s: float) -> float:
        return readout(y_f, s, self.base)

    def shift(self, y_f: np.ndarray | BaseVector, t: float) -> Vec:
        return shift(y_f, t, self.base)
