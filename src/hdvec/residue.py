"""Residue Holographic Computing (RHC) stubs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .fpe import generate_base, encode_fpe
from .core import bind
from .utils import phase_normalize
from .base import Vec


@dataclass
class ResidueEncoder:
    moduli: List[int]
    D: int

    def __post_init__(self) -> None:
        # One base per modulus
        self.bases = [generate_base(self.D) for _ in self.moduli]

    def __call__(self, x: int) -> np.ndarray:
        return encode_residue(x, self.moduli, np.stack(self.bases, axis=0))


def encode_residue(x: int, moduli: List[int], bases: np.ndarray) -> Vec:
    """Encode integer x under multiple moduli using FPE-style roots of unity."""
    if bases.shape[0] != len(moduli):
        raise ValueError("bases must have shape (len(moduli), D)")
    parts = []
    for k, m in enumerate(moduli):
        # Represent x mod m via exponentiation on the base
        r = float(x % m)
        z_m = encode_fpe(r, bases[k])
        parts.append(z_m)
    # Bundle across moduli
    v = np.sum(parts, axis=0).astype(np.complex64)
    return Vec(phase_normalize(v))


def res_add(a: np.ndarray, b: np.ndarray) -> Vec:
    """RHC add: phase-add via componentwise multiplication then renormalize."""
    return Vec(phase_normalize(a * b))


def res_mul(a: np.ndarray, b: np.ndarray) -> Vec:
    """RHC multiply: placeholder using hadamard bind (⋆)."""
    return bind(a, b, op="hadamard")


def crt_reconstruct(parts: np.ndarray, moduli: List[int]) -> int:
    """Chinese Remainder Theorem reconstruction for pairwise coprime moduli.

    Args:
        parts: Array of residues or integers; here we expect integers already reduced mod m.
        moduli: List of pairwise coprime moduli.
    Returns:
        The smallest non-negative solution modulo M = prod(moduli).
    """
    # Accept Python ints or small numpy ints
    residues = [int(parts[i]) % int(moduli[i]) for i in range(len(moduli))]
    Ms = 1
    for m in moduli:
        Ms *= int(m)

    def inv(a: int, m: int) -> int:
        # Modular inverse via extended Euclid
        t, new_t = 0, 1
        r, new_r = m, a % m
        while new_r != 0:
            q = r // new_r
            t, new_t = new_t, t - q * new_t
            r, new_r = new_r, r - q * new_r
        if r > 1:
            raise ValueError("a is not invertible")
        if t < 0:
            t += m
        return t

    x = 0
    for (ai, mi) in zip(residues, moduli):
        Mi = Ms // mi
        yi = inv(Mi, mi)
        x += ai * Mi * yi
    return int(x % Ms)
