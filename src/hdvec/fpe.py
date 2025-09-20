"""Fractional Power Encoding (FPE) bases and encoders.

Minimal stubs: unit-modulus bases and componentwise exponentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import phase_normalize


def generate_base(
    d: int,
    dist: Literal["uniform", "cauchy"] = "uniform",
    unitary: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a complex unit-modulus base vector of length D.

    Args:
        d: Dimensionality.
        dist: Phase distribution ("uniform" or "cauchy").
        unitary: If True, enforce unit modulus.
        rng: Optional RNG.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dist == "uniform":
        phases = rng.uniform(-np.pi, np.pi, size=d)
    elif dist == "cauchy":
        phases = np.arctan(rng.standard_cauchy(size=d))  # squashed tails
    else:
        raise ValueError(f"Unknown dist: {dist}")
    base = np.exp(1j * phases).astype(np.complex64)
    return phase_normalize(base) if unitary else base


def encode_fpe(r: float, base: np.ndarray) -> np.ndarray:
    """FPE: z(r) = base ** r (componentwise power).

    Assumes `base` is unit modulus. Result is complex64 phasor vector.
    """
    z = np.power(base, r)
    return phase_normalize(z)


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
