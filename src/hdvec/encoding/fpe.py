"""Fractional Power Encoding (FPE) bases and encoders.

Minimal stubs: unit-modulus bases and componentwise exponentiation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils import ensure_array, phase_normalize
from .kernels import estimate_kernel

__all__ = [
    "generate_base",
    "encode_fpe",
    "encode_fpe_vec",
    "make_circular_base",
    "encode_circular",
    "encode_boolean",
    "FPEEncoder",
    "estimate_kernel",
    "readout",
    "shift",
    "convolve",
    "probe",
]


def generate_base(
    d: int,
    dist: Literal["uniform", "gaussian", "laplace", "cauchy", "student", "mixture"] = "uniform",
    unitary: bool = True,
    rng: np.random.Generator | None = None,
    *,
    beta: float = 1.0,
    mixture_weights: Iterable[float] | None = None,
) -> np.ndarray:
    """Generate a complex unit-modulus base vector of length D.

    Args:
        d: Dimensionality.
        dist: Phase distribution.
        unitary: If True, enforce unit modulus.
        rng: Optional RNG.
        beta: Bandwidth parameter controlling spread.
        mixture_weights: Optional weights for mixture distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    phases = _sample_phases(dist, d, rng, beta=beta, mixture_weights=mixture_weights)
    base = np.exp(1j * phases).astype(np.complex64)
    return phase_normalize(base) if unitary else base


def _sample_phases(
    dist: str,
    size: int,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
    mixture_weights: Iterable[float] | None = None,
) -> np.ndarray:
    if dist == "uniform":
        return rng.uniform(-np.pi, np.pi, size=size)
    if dist == "gaussian":
        return rng.normal(loc=0.0, scale=beta, size=size)
    if dist == "laplace":
        return rng.laplace(loc=0.0, scale=beta, size=size)
    if dist == "cauchy":
        return np.arctan(rng.standard_cauchy(size=size) * beta)
    if dist == "student":
        return rng.standard_t(df=max(beta, 1.0), size=size)
    if dist == "mixture":
        if mixture_weights is None:
            raise ValueError("mixture distribution requires mixture_weights")
        weights = np.asarray(list(mixture_weights), dtype=float)
        weights = weights / weights.sum()
        components = [rng.normal(0.0, beta * (i + 1), size=size) for i in range(len(weights))]
        samples = np.zeros(size, dtype=float)
        for w, comp in zip(weights, components, strict=False):
            samples += w * comp
        return samples
    raise ValueError(f"Unknown dist: {dist}")


def encode_fpe(r: float, base: np.ndarray) -> np.ndarray:
    """FPE: z(r) = base ** r (componentwise power).

    Assumes `base` is unit modulus. Result is complex64 phasor vector.
    """
    z = np.power(base, r)
    return phase_normalize(z)


def encode_fpe_vec(x: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
    """Encode a vector x ∈ R^n by separably binding per-axis encodings.

    Returns the Hadamard product of per-axis encodings: ⊛_k base_k^(x_k).
    """
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if len(bases) != x.size:
        raise ValueError("len(bases) must equal x.size")
    out = np.ones_like(bases[0], dtype=np.complex64)
    for k, b in enumerate(bases):
        out = out * encode_fpe(float(x[k]), b)
    return out.astype(np.complex64)


def readout(y_f: np.ndarray, s: float, base: np.ndarray) -> float:
    """Evaluate the represented function at ``s`` via inner product."""
    y = ensure_array(y_f).astype(np.complex64)
    probe = encode_fpe(s, base)
    denom = max(1, probe.size)
    return float((np.conj(probe) * y).sum().real / denom)


def shift(y_f: np.ndarray, delta: float, base: np.ndarray) -> np.ndarray:
    """Shift the represented function by ``delta`` using binding."""
    return encode_fpe(delta, base) * ensure_array(y_f)


def convolve(y_f: np.ndarray, y_g: np.ndarray) -> np.ndarray:
    """Convolve two function representations via elementwise multiply."""
    return ensure_array(y_f) * ensure_array(y_g)


def probe(y_f: np.ndarray, xs: Iterable[float], base: np.ndarray) -> np.ndarray:
    """Evaluate ``y_f`` at multiple points specified in ``xs``."""
    values = [readout(y_f, float(s), base) for s in xs]
    return np.asarray(values, dtype=np.float64)


def make_circular_base(d: int, period: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Make a circular (period-L) base with phases sampled on the L-point grid.

    Each component phase is 2π*j/L for a random integer j ∈ {0..L-1}.
    """
    if period <= 0:
        raise ValueError("L must be positive")
    if rng is None:
        rng = np.random.default_rng()
    js = rng.integers(0, period, size=d)
    angles = (2.0 * np.pi * js / float(period)).astype(np.float64)
    return np.asarray(np.exp(1j * angles).astype(np.complex64))


def encode_circular(r: float, period: int, base_period: np.ndarray) -> np.ndarray:
    """Encode a circular value with period L: p(r) = base^(r), p(r+L) = p(r).

    Works for real or fractional r; periodicity holds with period L.
    """
    # Raising base to power r implicitly scales phase; periodicity follows from base roots.
    out = (base_period**r).astype(np.complex64)
    return np.asarray(out)


def encode_boolean(bit: int, base_l2: np.ndarray) -> np.ndarray:
    """Encode a Boolean as L=2 circular value: 0 -> ones, 1 -> base_L2.

    ``base_L2`` should be constructed by ``make_circular_base(D, L=2)``.
    """
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    return np.ones_like(base_l2) if bit == 0 else base_l2.astype(np.complex64)


@dataclass
class FPEEncoder:
    """Encoder for FPE given a base vector.

    Example:
        enc = FPEEncoder(D=1024)
        z = enc(1.5)
    """

    D: int
    dist: Literal["uniform", "cauchy"] = "uniform"
    unitary: bool = True
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.base = generate_base(self.D, dist=self.dist, unitary=self.unitary, rng=self.rng)

    def __call__(self, r: float) -> np.ndarray:
        return encode_fpe(r, self.base)


def learn_fpe_phases(encoder: FPEEncoder, data: np.ndarray, optimizer: str = "sgd") -> None:
    """Placeholder for phase learning. No-op in scaffolding.

    Args:
        encoder: FPEEncoder instance.
        data: Training data array.
        optimizer: Strategy (unused).
    """
    _ = (encoder, data, optimizer)
    return None
