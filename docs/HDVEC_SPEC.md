# HDVEC — Vision, Scope, and V1 Specification

This document defines the goals, scope, APIs, and configuration for the initial release (V1) of the `hdvec` library. It is intended as a reference for engineering implementation and review.

HDVEC provides a high‑quality, NumPy‑first, optionally‑JITed core for Hyperdimensional / Vector Symbolic Architectures (HDC/VSA) focused on complex‑phasor domains (HRR/FHRR family), with foundational encoders (FPE/VFA) and stubs for GHRR and Residue HDC. It is general‑purpose but designed to serve as an “engine” within the broader HOLOGRAM architecture.

## Goals (V1)

- Deliver a small, composable core that implements the canonical HDC/VSA algebra and selected encoders:
  - Vector operations: bundling, binding, permutation, similarity.
  - Encoders: FPE (Fractional Power Encoding), VFA (function encoders), residue (RHC) stubs, GHRR stubs.
  - Decoding utilities: simple matching; CRT for residue.
- NumPy as the primary backend, with optional Numba JIT acceleration that is a no‑op when Numba is absent.
- Clear, typed Python API with light object wrappers where additive value exists (e.g., `Vec`, `ResidueEncoder`, `VFAEncoder`, `FPEEncoder`).
- Robustness and repeatability: deterministic RNG seeding options, invariants (unit‑modulus where appropriate), and test coverage.
- Developer ergonomics: ruff/black/mypy pre‑commit compliance; Sphinx docs with MyST (Markdown) pages; CI across Python 3.10–3.12.

## Non‑Goals (V1)

- No SDM (Sparse Distributed Memory), FOAM/resonator engines, FSA, or SSM in this package (will live in sibling libraries).
- No GPU/CUDA or Torch backends in V1 (leave adapters/experiments for V1.x+). Keep public API neutral to allow future backends.
- No binary/sparse domains initially. V1 focuses on complex‑phasor vectors; binaries can be introduced later in a parallel subpackage.
- No heavy serialization/story around codebook stores (basic RNG seeding and arrays suffice for V1).

## Architecture Overview

Top‑level package: `hdvec`

- `hdvec.core` — core algebra: `bind`, `bundle`, `similarity`, `permute` (NumPy; optional Numba kernel for hot paths).
- `hdvec.fpe` — Fractional Power Encoding; base generation; scalar encoding; encoder class.
- `hdvec.vfa` — VFA helpers to encode functions and readout/shift/convolve; `VFAEncoder` thin wrapper.
- `hdvec.residue` — Residue HDC encoders and basic arithmetic bindings; CRT utilities.
- `hdvec.ghrr` — GHRR sampling and operations (stubs in V1; incremental filling allowed).
- `hdvec.decoding` — simple decoding/readout helpers (anchors, argmax readout, etc.).
- `hdvec.utils` — utilities: `optional_njit`, `ensure_array`, `phase_normalize`, `inject_noise`, etc.
- `hdvec.config` — mutable global `Config` with conservative, typed fields.
- `hdvec.base` — `BaseVector` ABC and `Vec` convenience wrapper around `np.ndarray`.

Unit tests live in `tests/` mirroring modules. Docs in `docs/` (Sphinx + MyST).

## Data Types & Invariants

- Default dtype: `complex64` (phasor vectors), enforced via normalization paths.
- Vector shape: 1‑D arrays of length `D` unless stated otherwise; GHRR may use `(D, m, m)` slices.
- Unit‑modulus invariant for phasor operations when relevant; `phase_normalize` projects back.
- `Vec` wraps a NumPy array and conforms to `BaseVector` for transitional APIs (accepts either `np.ndarray` or `BaseVector`, returns `Vec`).

## Configuration (Global)

Module: `hdvec.config`

```python
from dataclasses import dataclass
from typing import Literal, Any
import numpy as np

@dataclass
class Config:
    D: int = 1024                                 # default dimensionality
    backend: Literal["numpy", "torch"] = "numpy"  # future‑proofing
    dtype: np.dtype[np.complexfloating[Any, Any]] = np.dtype(np.complex64)
    binding: Literal["hadamard", "cc", "lcc"] = "hadamard"  # default bind op
    dist: Literal["uniform", "cauchy"] = "uniform"           # FPE base distribution
    m: int = 1                                      # GHRR slice size (m×m)
    moduli: list[int] | None = None                # Residue moduli
    conv_backend: Literal["fft"] = "fft"          # convolution back‑end

def get_config() -> Config: ...
```

Notes:
- Expose `get_config()` only in V1; set via attribute assignment (simple and explicit). Avoid dynamic context stacks until needed.
- Leave RNG control to callers (functions accept `rng: np.random.Generator | None`).

## Core Algebra (hdvec.core)

Signatures (typed):

```python
from typing import Protocol
import numpy as np

class BaseVector(Protocol):
    @property
    def data(self) -> np.ndarray: ...

def bind(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector, op: str = "hadamard") -> Vec
def bundle(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec
def similarity(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> float
def permute(v: np.ndarray | BaseVector, shift: int) -> Vec
```

Behavior:
- `hadamard` (default): elementwise multiply; phase‑normalize if complex.
- `cc`: circular convolution via FFT; `ifft(fft(a) * fft(b))`; cast to `complex64` if either operand is complex; else take `real`.
- `lcc`: not implemented in V1 (placeholder).
- `bundle`: elementwise sum then normalize: phasor → `phase_normalize(out)`, real → average.
- `similarity`: real part of normalized inner product (Numba‑accelerated path when available).

## FPE (hdvec.fpe)

Purpose: Encode scalars (and small vectors) via fractional power of a phasor base.

API:

```python
def generate_base(d: int,
                  dist: Literal["uniform", "cauchy"] = "uniform",
                  unitary: bool = True,
                  rng: np.random.Generator | None = None) -> np.ndarray

def encode_fpe(x: float, base: np.ndarray) -> np.ndarray  # returns complex64 phasor vector

@dataclass
class FPEEncoder:
    D: int
    dist: Literal["uniform", "cauchy"] = "uniform"
    unitary: bool = True
    def __call__(self, x: float) -> np.ndarray: ...
```

Behavior:
- `generate_base`: draw unit‑modulus phasor entries; distribution controls phase draw.
- `encode_fpe`: returns `base ** x` in the phasor sense (exp(i * angle(base) * x)).

## VFA (hdvec.vfa)

Purpose: Vector Function Architecture utilities: represent functions via FPE bases and vector algebra.

API:

```python
def encode_function(points: np.ndarray, alphas: np.ndarray, base: np.ndarray) -> Vec
def readout(y_f: np.ndarray | BaseVector, s: float, base: np.ndarray) -> float
def shift(y_f: np.ndarray | BaseVector, t: float, base: np.ndarray) -> Vec
def convolve(y_f: np.ndarray | BaseVector, y_g: np.ndarray | BaseVector) -> Vec

@dataclass
class VFAEncoder:
    base: np.ndarray
    def encode(self, points: np.ndarray, alphas: np.ndarray) -> Vec
    def readout(self, y_f: np.ndarray | BaseVector, s: float) -> float
    def shift(self, y_f: np.ndarray | BaseVector, t: float) -> Vec
```

Notes:
- `readout(y_f, s)` ≈ `similarity(y_f, encode_fpe(s, base))`.
- `shift(y_f, t)` ≈ `bind(y_f, encode_fpe(t, base))`.
- `convolve` implemented via FFT as a placeholder for function convolution in the VFA view.

## Residue HDC (hdvec.residue)

Purpose: Stubs and minimal utilities for Residue HDC encodings and arithmetic; CRT reconstruction. Full resonator/FOAM demixing is out of scope in V1.

API:

```python
@dataclass
class ResidueEncoder:
    moduli: list[int]
    D: int
    def __call__(self, x: int) -> Vec

def encode_residue(x: int, moduli: list[int], bases: np.ndarray) -> Vec
def res_add(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec  # phasor add via Hadamard
def res_mul(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec  # placeholder via bind
def crt_reconstruct(parts: np.ndarray, moduli: list[int]) -> int
```

Behavior:
- One FPE base per modulus (`bases` shape `(K, D)`); encode each residue and bundle (sum→normalize) or multiply directly if aligning with RHC conventions.
- `res_add` uses Hadamard + phase normalization; `res_mul` currently maps to `bind(..., op="hadamard")` as a placeholder.
- `crt_reconstruct` implements Chinese Remainder Theorem for pairwise coprime moduli.

## GHRR (hdvec.ghrr)

Purpose: Stubs to sample GHRR vectors and compute similarities/binds; mature in V1.x.

API (subset):

```python
@dataclass
class GHVec:
    data: np.ndarray  # shape (D, m, m), complex64
    @property
    def D(self) -> int: ...
    @property
    def m(self) -> int: ...

def sample_ghrr(d: int, m: int, policy: str = "haar", rng: np.random.Generator | None = None) -> GHVec
def gh_similarity(a: GHVec, b: GHVec) -> float
def gh_bind(a: GHVec, b: GHVec) -> GHVec
```

## Decoding (hdvec.decoding)

Purpose: Simple helpers for nearest‑anchor decoding and toy resonator placeholders.

API:

```python
@dataclass
class Anchors:
    anchors: np.ndarray  # shape (K, D)

def decode_point(y: np.ndarray, anchors: np.ndarray) -> int  # argmax similarity
```

## Utilities (hdvec.utils)

Key functions:

```python
def optional_njit(*args, **kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]  # no‑op if numba missing
def ensure_array(x: np.ndarray | Any) -> np.ndarray
def phase_normalize(v: np.ndarray) -> np.ndarray
def inject_noise(v: np.ndarray, sigma: float, dist: str = "vonmises", rng: np.random.Generator | None = None) -> np.ndarray
```

## Error Handling & Contracts

- Validate shapes and argument consistency (e.g., matching shapes for `encode_function`).
- Raise `ValueError` for unsupported `op` or mismatched shapes.
- Keep operations pure w.r.t inputs; `Vec.normalize()` mutates its instance by design but core functional APIs return new arrays/Vecs.

## Performance & Backends

- Use `optional_njit` to provide Numba kernels for hotspots (`similarity`, potential future ones) without adding a hard dependency.
- Keep FFT ops via NumPy; allow swapping FFT backend through config in future if needed.
- Plan for Torch/JAX adapters in V1.x without committing to their presence in V1.

## Testing (V1 must‑have)

- Algebraic identities: bind/unbind behavior, permutation invariants, similarity numeric ranges.
- FPE/VFA: `encode_fpe(x+y) ≈ bind( fpe(x), fpe(y) )`, readout accuracy sanity check.
- Residue: shape checks, CRT correctness on simple inputs, `res_add` phasor unit‑modulus.
- GHRR: sampling shapes and basic similarity tests.
- Utilities: `phase_normalize` unit‑magnitude, `inject_noise` preserves magnitude.

## Documentation

- MyST‑Markdown docs; link this spec from README and docs index in V1.
- Minimal examples (`examples/`) demonstrating FPE/VFA readout and residue encode/CRT.

## CI / Tooling

- Lint: ruff (E, F, I, N, UP, B, A, C90). Format: black (line‑length 100). Types: mypy strict-ish.
- Matrix: 3.10/3.11/3.12; optional numba job to verify importability.

## Roadmap (Post‑V1)

- Residue HDC: resonator & FOAM demixing engines; multiplicative binding operator; hexagonal lattices for 2D positioning.
- Backends: Torch/JAX; GPU FFTs; batching.
- Domains: binary and sparse HVs under a parallel namespace (e.g., `hdvec.binary`).
- Serialization & registry: codebooks, permutations, reproducible runs.
- SDM/SSM/FSA: separate packages that consume HDVEC core.

## Out‑of‑Scope for HDVEC (lives in sibling libs)

- SDM variants and associative memory stores.
- FOAM/resonator implementations beyond toy examples.
- Finite‑state automata; SSM layers; attention and relation encoding.

---

This spec reflects the current scaffold and refactor direction already present in the repository (NumPy‑first, optional Numba, typed APIs) and sets a concrete V1 surface area that is broadly useful beyond HOLOGRAM while staying lean.

