# HDVEC Post‑V1 Roadmap

This document outlines suggested enhancements after V1, with rationale and likely design directions. These items are not required to consume HDVEC in HOLOGRAM but will broaden capability and performance.

## 1) GHRR Operators & Interop

- Structured families: block‑circulant / banded unitary slices; parameterized sampling for controllable operator spectra.
- Interop with Vec: expose `apply(GHVec, v: Vec)` to act as a per‑slice operator over vectorized views; support composition/inversion.
- Unit tests: operator closure, adjoint/inverse identities, stability under bundling.

## 2) Residue HDC Extensions

- Dedicated exponentiation‑binding interface (⋆): standardized API to express multiplicative composition given decoded scalars or per‑modulus residues.
- FOAM/resonator demixer: provided as a sibling library (e.g., `hdfoam`) with clean hooks in HDVEC (codebook builders, similarity kernels).
- Hexagonal residue lattices for 2D coordinates; helpers to project between square and hex systems.

## 3) Backends & Performance

- Torch/JAX adapters: thin shims that mirror `hdvec.core` signatures on tensors; optional GPU FFTs.
- Batched algebra: broadcast‑friendly bind/bundle `(N,D)` x `(N,D)` and `(N,D)` x `(D,)` with minimal copies.
- FFT backend switches: pocketfft/pyFFTW under a small strategy object when FFT perf dominates.

## 4) Tooling & Packaging

- Benchmarks: scripts to chart ops/sec vs `D` and batch sizes; compare CPU vs GPU paths when adapters are enabled.
- Docs site: Sphinx + MyST publishing (gh‑pages or ReadTheDocs); API examples and quickstarts.
- Versioning: move from 0.1.x (alpha) to 0.2.x once slow tests and doc polish land; publish to PyPI when downstream HOLOGRAM interfaces stabilize.

