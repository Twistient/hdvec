"""Residue Holographic Computing (RHC).

Implements multi-modulus residue encoders using FPE bases, addition via phasor
Hadamard multiplication, and utilities for multiplicative operations and
decoding via a lightweight resonator. See Kymn et al. (2023).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..base import BaseVector, Vec
from ..utils import ensure_array, phase_normalize
from .fpe import encode_fpe, generate_base


@dataclass
class ResidueBases:
    """Reusable container for residue base vectors and codebooks."""

    moduli: list[int]
    stack: np.ndarray  # shape (K, D)
    _codebooks: list[np.ndarray] | None = None

    @property
    def D(self) -> int:  # noqa: N802
        return int(self.stack.shape[1])

    @property
    def codebooks(self) -> list[np.ndarray]:
        if self._codebooks is None:
            self._codebooks = build_codebooks(self.moduli, self.stack)
        return self._codebooks

    @classmethod
    def from_moduli(
        cls,
        moduli: Sequence[int],
        D: int,  # noqa: N803
        *,
        rng: np.random.Generator | None = None,
    ) -> ResidueBases:
        if rng is None:
            rng = np.random.default_rng()
        bases = [generate_base(D, rng=rng) for _ in moduli]
        stack = np.stack(bases, axis=0)
        return cls(list(moduli), stack)


@dataclass
class ResidueEncoder:
    moduli: list[int]
    D: int

    def __post_init__(self) -> None:
        self.bases = ResidueBases.from_moduli(self.moduli, self.D)

    def __call__(self, x: int) -> Vec:
        return encode_residue(x, self.bases)


def encode_residue(x: int, bases: ResidueBases | np.ndarray, moduli: Iterable[int] | None = None) -> Vec:
    """Encode integer ``x`` using residue bases."""
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    parts = []
    for k, m in enumerate(mods):
        r = float(x % m)
        z_m = encode_fpe(r, stack[k])
        parts.append(z_m)
    v = np.sum(parts, axis=0).astype(np.complex64)
    return Vec(phase_normalize(v))


def res_add(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """RHC add: phase-add via componentwise multiplication then renormalize."""
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    result = phase_normalize(a_arr * b_arr)
    return Vec(result)


def res_pow_scalar(
    a: np.ndarray | BaseVector,
    p: int,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
) -> Vec:
    """Multiply an encoded residue vector by integer ``p`` (i.e., encode x*p).

    Implementation decodes per-modulus residues via a light resonator, applies
    the modular multiplication r_k' = (r_k * p) mod m_k, then re-bundles.
    """
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    residues = resonator_decode(a, stack, mods)
    codebooks = build_codebooks(mods, stack)
    parts = []
    for j, m in enumerate(mods):
        idx = (int(residues[j]) * int(p)) % int(m)
        parts.append(codebooks[j][idx])
    v = np.sum(parts, axis=0).astype(np.complex64)
    return Vec(phase_normalize(v))


def res_mul_int(
    a: int,
    b: int,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
) -> Vec:
    """Encode product of integers under the residue encoder: z(a*b).

    Convenience for callers that have raw integers and the encoder's bases.
    """
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    return encode_residue(int(a) * int(b), stack, mods)


def build_codebooks(moduli: list[int], bases: np.ndarray) -> list[np.ndarray]:
    """Build per-modulus codebooks Z_k with rows z_k(r), r=0..m_k-1."""
    zks: list[np.ndarray] = []
    for k, m in enumerate(moduli):
        rows = [encode_fpe(float(r), bases[k]) for r in range(int(m))]
        zks.append(np.stack(rows, axis=0).astype(np.complex64))
    return zks


def residue_correlations(
    v: np.ndarray | BaseVector,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
) -> list[np.ndarray]:
    """Return per‑modulus correlation scores against each codebook row.

    For each modulus m_k, returns a 1‑D array of shape (m_k,) with real scores
    (cosine-like) computed as (Z_k @ conj(v)).real.
    """
    vec = ensure_array(v)
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    codebooks = build_codebooks(mods, stack)
    scores: list[np.ndarray] = []
    for zk in codebooks:
        scores.append((zk @ np.conj(vec)).real.astype(np.float64))
    return scores


def residue_initial_guess(
    v: np.ndarray | BaseVector,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
) -> list[int]:
    """Return argmax indices per modulus as an initial residue estimate."""
    sc = residue_correlations(v, bases, moduli)
    return [int(np.argmax(s)) for s in sc]


def resonator_decode(
    v: np.ndarray | BaseVector,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
    steps: int = 16,
) -> list[int]:
    """Factor a residue-encoded vector into per-modulus residues via a simple resonator.

    Iteratively updates each factor by matching against its codebook while
    holding others fixed. Returns the list of residues r_k in [0, m_k).
    """
    vec = ensure_array(v)
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    codebooks = build_codebooks(mods, stack)
    # Initialize by direct correlation
    indices = [int(np.argmax((zk @ np.conj(vec)).real)) for zk in codebooks]
    # Iteratively refine by explain-away subtraction
    for _ in range(3):
        changed = False
        for j, zk in enumerate(codebooks):
            recon_others = np.zeros_like(vec)
            for i, idx in enumerate(indices):
                if i == j:
                    continue
                recon_others += codebooks[i][idx]
            residual = vec - recon_others
            scores = (zk @ np.conj(residual)).real
            new_idx = int(np.argmax(scores))
            changed = changed or (new_idx != indices[j])
            indices[j] = new_idx
        if not changed:
            break
    return indices


def res_decode_int(
    v: np.ndarray | BaseVector,
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None = None,
) -> int:
    """Decode a residue-encoded vector to an integer via resonator + CRT."""
    stack, mods = _resolve_bases_and_moduli(bases, moduli)
    parts = resonator_decode(v, stack, mods)
    return crt_reconstruct(np.array(parts, dtype=int), mods)


def _resolve_bases_and_moduli(
    bases: ResidueBases | np.ndarray,
    moduli: Iterable[int] | None,
) -> tuple[np.ndarray, list[int]]:
    if isinstance(bases, ResidueBases):
        mods = bases.moduli
        stack = bases.stack
        if moduli is not None and list(moduli) != mods:
            raise ValueError("Provided moduli do not match ResidueBases")
        return stack, mods
    stack = np.asarray(bases)
    if moduli is None:
        raise ValueError("moduli must be provided when passing raw base array")
    mods_list = list(moduli)
    if stack.shape[0] != len(mods_list):
        raise ValueError("bases must have shape (len(moduli), D)")
    return stack, mods_list


def crt_reconstruct(parts: np.ndarray, moduli: list[int]) -> int:
    """Chinese Remainder Theorem reconstruction for pairwise coprime moduli.

    Args:
        parts: Array of residues or integers; here we expect integers already reduced mod m.
        moduli: List of pairwise coprime moduli.
    Returns:
        The smallest non-negative solution modulo M = prod(moduli).
    """
    # Accept Python ints or small numpy ints
    residues = [int(parts[i]) % int(moduli[i]) for i in range(len(moduli))]
    ms = 1
    for m in moduli:
        ms *= int(m)

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
    for ai, mi in zip(residues, moduli, strict=False):
        mi_big = ms // mi
        yi = inv(mi_big, mi)
        x += ai * mi_big * yi
    return int(x % ms)
