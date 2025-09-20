<div align="center">

# HDVEC

Hyperdimensional Vector Computing for Python

</div>

HDVEC is a NumPy‑first, typed library for Hyperdimensional/Vector Symbolic Architectures (HDC/VSA). It provides the algebra, encoders, and utilities needed to build robust, compositional systems in high dimensions. HDVEC is designed to be general‑purpose and also serve as a core engine for HOLOGRAM (a neurosymbolic reasoning architecture).

Status: alpha (APIs may evolve). Python 3.10–3.12; optional Numba acceleration.

## Features

- Algebra (phasor HRR/FHRR)
  - bind, unbind/inv (Hadamard/conjugate), bundle, similarity/cosine, permutation (roll)
  - projection to unit modulus (phase‑only) to arrest drift
  - HRR interop: FFT circular convolution/correlation
- Encoders
  - FPE (Fractional Power Encoding): real/circular/Boolean values; multi‑dimensional FPE
  - VFA helpers (Vector Function Architecture): readout, shift, convolution
  - Grid/field encoder: build scene hypervectors from per‑cell positions and values
  - Residue HDC (RNS overlay): residue encoders across co‑prime moduli, addition via phasor multiply, simple per‑modulus decode + CRT
  - GHRR (operator‑valued): per‑dimension unitary slices; bind (matmul), bundle (QR), similarity (trace)
- Cleanup & batch
  - Codebook utilities, cosine top‑k/nearest (single and batch)
- Engineering
  - Typed API, py.typed, tests, CI, Sphinx docs (MyST), pre‑commit hooks

## Install

Using uv (recommended):

- Create a virtual environment (Python 3.10+)
  - `uv venv -p 3.12`
- Activate it
  - `source .venv/bin/activate`
- Install (editable) with dev extras
  - `uv pip install -e ".[dev]"`
- Optional extras
  - Numba: `uv pip install -e ".[dev,numba]"`
  - Torch: `uv pip install -e ".[torch]"`
  - Docs: `uv pip install -e ".[docs]"`

Or with pip: `pip install -e ".[dev]"` (ensure Python 3.10+).

## Quick Examples

Binding and FPE

```python
import numpy as np
from hdvec import bind, unbind
from hdvec.fpe import FPEEncoder, generate_base

D = 1024
base = generate_base(D)
enc = FPEEncoder(D=D)

z1 = enc(0.7)
z2 = enc(1.3)
z_sum = bind(z1, z2)           # enc(0.7 + 1.3)
z1_rec = unbind(z_sum, z2)     # ≈ z1
```

Grid/Field encoding (VFA)

```python
from hdvec.vfa import encode_grid, read_cell, translate_grid
K = 5
codebook = np.stack([np.exp(1j*np.random.uniform(-np.pi,np.pi,size=D)).astype(np.complex64)
                     for _ in range(K)], axis=0)
values = (np.arange(16) % K).reshape(4,4)
scene = encode_grid(values, [base, base], value_codebook=codebook)
idx, score = read_cell(scene, 1, 2, [base, base], value_codebook=codebook)
scene2 = translate_grid(scene, 1.0, 0.0, [base, base])
```

Residue HDC (RNS overlay)

```python
from hdvec.residue import ResidueEncoder, res_pow_scalar, res_decode_int
moduli = [3, 5]
enc_res = ResidueEncoder(moduli=moduli, D=D)
vx = enc_res(7)
# multiply by integer k=2 via per‑modulus decode + rebundle
vx2 = res_pow_scalar(vx, 2, moduli, np.stack(enc_res.bases, axis=0))
# decode to integer via per‑modulus residues + CRT
x_rec = res_decode_int(vx2, moduli, np.stack(enc_res.bases, axis=0))
```

GHRR

```python
from hdvec.ghrr import sample_ghrr, gh_bind, gh_unbind, gh_project_unitary
G1 = sample_ghrr(8, 2)
G2 = sample_ghrr(8, 2)
G12 = gh_bind(G1, G2)
G1_rec = gh_unbind(G12, G2)
G12_u = gh_project_unitary(G12)
```

More examples: see `docs/quickstart_grid.md`.

## Use Cases

- Structure‑aware perception and reasoning in high dimension
- Compositional encoding of sets, sequences, records, scenes
- Numeric/circular attributes (FPE) with translation‑equivariant ops (VFA)
- Large‑range integers and cycles (Residue HDC)
- Operator‑valued encodings and matrix‑like bindings (GHRR)

## Design & Performance

- NumPy‑first with optional Numba JIT (no‑op fallback); `complex64` default
- Recommended D: 8k–32k for concentration effects (see docs/HDVEC_SCALING.md)
- CI: Python 3.10/3.11/3.12; lint (ruff), format (black), types (mypy), tests (pytest)

## References

- Plate, T. (1995). Holographic Reduced Representations (HRR).
- Kleyko, D., et al. (2022). A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I.
- Frady, E. P., et al. (2021). Computing on Functions Using Randomized Vector Representations (Vector Function Architecture).
- Verges, C., et al. (2025). Learning encoding phasors with Fractional Power Encoding.
- Kymn, C., et al. (2023). Computing with Residue Numbers in High‑Dimensional Representation.
- Yeung, S., et al. (2024). Generalized Holographic Reduced Representations.

(See `references/` for PDFs and notes.)

## Contributing

- Pre‑commit hooks: `pre-commit install` then `pre-commit run --all-files`
- Style: ruff + black; types: mypy; tests: pytest
- Please open issues/PRs with repro steps and concise diffs.

## Versioning & Releases

- Semantic versioning.
- Current version: 0.1.0 (alpha). For initial release, cut a GitHub Release from `master` with changelog.
- Maintain a simple CHANGELOG.md for 0.1.x → 0.2.x once slow tests & docs polish land.

## License

See LICENSE.
