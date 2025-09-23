"""Core algebraic operations for hyperdimensional vectors."""

from __future__ import annotations

import numpy as np

from ..base import BaseVector, Vec
from ..config import get_config
from ..errors import BundlingModeError, ConfigurationError, InvalidBindingError, ShapeMismatchError
from ..utils import ensure_array, phase_normalize

__all__ = [
    "bind",
    "inv",
    "unbind",
    "bundle",
    "permute",
    "project_unitary",
    "circ_conv",
    "circ_corr",
]


def bind(
    a: np.ndarray | BaseVector,
    b: np.ndarray | BaseVector,
    *,
    op: str = "hadamard",
) -> Vec:
    """Bind two hypervectors using the selected operator."""
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    if op == "hadamard":
        out = a_arr * b_arr
        out = phase_normalize(out) if np.iscomplexobj(out) else out
        return Vec(out)
    if op == "cc":
        fa = np.fft.fft(a_arr)
        fb = np.fft.fft(b_arr)
        out = np.fft.ifft(fa * fb)
        out = (
            out.astype(np.complex64)
            if np.iscomplexobj(a_arr) or np.iscomplexobj(b_arr)
            else out.real
        )
        return Vec(out)
    if op == "lcc":
        return _localized_circular_conv(a_arr, b_arr)
    raise InvalidBindingError(f"Unknown binding op: {op}")


def inv(a: np.ndarray | BaseVector) -> Vec:
    """Return the algebraic inverse of ``a``."""
    arr = ensure_array(a)
    if np.iscomplexobj(arr):
        return Vec(np.conj(arr))
    return Vec(arr)


def unbind(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector, *, op: str = "hadamard") -> Vec:
    """Unbind ``a`` with ``b`` using the provided binding operator."""
    return bind(a, inv(b), op=op)


def bundle(  # noqa: C901
    *vectors: np.ndarray | BaseVector,
    normalize: str = "phasor",
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> Vec:
    """Superpose one or more vectors with configurable normalization."""

    if not vectors:
        raise BundlingModeError("bundle() requires at least one vector")

    arrays = [ensure_array(v) for v in vectors]
    reference_shape = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != reference_shape:
            raise ShapeMismatchError("All vectors must share the same shape for bundling")

    stack = np.stack(arrays, axis=0)
    stack_dtype = stack.dtype

    if weights is not None:
        weights_arr = np.asarray(weights)
        if weights_arr.shape != (stack.shape[0],):
            raise ShapeMismatchError("weights must be a 1-D array matching the number of vectors")
        weights_arr = weights_arr.astype(stack_dtype, copy=False)
        accum = np.tensordot(weights_arr, stack, axes=(0, 0))
        weight_sum = np.sum(weights_arr.real)
    else:
        accum = stack.sum(axis=0)
        weight_sum = float(stack.shape[0])

    normalize_key = normalize.lower()
    is_complex = np.iscomplexobj(accum)

    if normalize_key == "phasor":
        if is_complex:
            out = phase_normalize(accum)
        else:
            denom = weight_sum if abs(weight_sum) > 1e-12 else 1.0
            out = accum / denom
    elif normalize_key == "mean":
        denom = weight_sum if abs(weight_sum) > 1e-12 else 1.0
        out = accum / denom
    elif normalize_key == "l2":
        norm = float(np.linalg.norm(accum))
        out = accum if norm == 0.0 else accum / norm
    elif normalize_key == "none":
        out = accum
    else:
        raise BundlingModeError(f"Unknown bundling normalization: {normalize}")

    if is_complex:
        return Vec(np.asarray(out, dtype=np.complex64))
    return Vec(np.asarray(out, dtype=arrays[0].dtype))


def permute(v: np.ndarray | BaseVector, shift: int) -> Vec:
    """Cyclically permute along the last axis."""
    v_arr = ensure_array(v)
    return Vec(np.roll(v_arr, shift, axis=-1))


def project_unitary(v: np.ndarray | BaseVector) -> Vec:
    """Project onto the unit circle (phase-only) or normalize real vectors."""
    arr = ensure_array(v)
    if np.iscomplexobj(arr):
        angles = np.angle(arr)
        return Vec(np.exp(1j * angles).astype(np.complex64))
    norm = float(np.linalg.norm(arr))
    return Vec(arr if norm == 0.0 else (arr / norm).astype(arr.dtype))


def circ_conv(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Circular convolution via FFT."""
    aa = ensure_array(a)
    bb = ensure_array(b)
    fa = np.fft.fft(aa)
    fb = np.fft.fft(bb)
    out = np.fft.ifft(fa * fb)
    return Vec(
        out.astype(np.complex64) if (np.iscomplexobj(aa) or np.iscomplexobj(bb)) else out.real
    )


def circ_corr(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Circular correlation (deconvolution) via FFT."""
    aa = ensure_array(a)
    bb = ensure_array(b)
    fa = np.fft.fft(aa)
    fb = np.fft.fft(bb)
    out = np.fft.ifft(fa * np.conj(fb))
    return Vec(
        out.astype(np.complex64) if (np.iscomplexobj(aa) or np.iscomplexobj(bb)) else out.real
    )


def _localized_circular_conv(a_arr: np.ndarray, b_arr: np.ndarray) -> Vec:
    cfg = get_config()
    blocks = cfg.lcc_blocks
    if blocks is None:
        raise ConfigurationError(
            "Config.lcc_blocks is None; set a positive integer before using lcc binding."
        )
    if blocks <= 0:
        raise ConfigurationError("Config.lcc_blocks must be positive for lcc binding")
    if a_arr.shape != b_arr.shape:
        raise ShapeMismatchError("lcc binding requires operands with matching shapes")
    last_dim = a_arr.shape[-1]
    if last_dim % blocks != 0:
        raise ShapeMismatchError(
            f"Trailing dimension {last_dim} must be divisible by lcc_blocks={blocks}"
        )
    block_len = last_dim // blocks
    prefix = int(np.prod(a_arr.shape[:-1])) if a_arr.ndim > 1 else 1
    a_blocks = a_arr.reshape(prefix, blocks, block_len)
    b_blocks = b_arr.reshape(prefix, blocks, block_len)
    fa = np.fft.fft(a_blocks, axis=-1)
    fb = np.fft.fft(b_blocks, axis=-1)
    out = np.fft.ifft(fa * fb, axis=-1).reshape(a_arr.shape)
    if np.iscomplexobj(a_blocks) or np.iscomplexobj(b_blocks):
        return Vec(out.astype(np.complex64))
    return Vec(out.real)
