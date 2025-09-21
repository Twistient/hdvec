<div align="center">

# HDVEC

Hyperdimensional Vector Computing for Python

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)
[![License: LGPLv3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](LICENSE)

</div>

HDVEC is a NumPy‑first, typed library for Hyperdimensional/Vector Symbolic Architectures (HDC/VSA). It provides the algebra, encoders, and utilities needed to build robust, compositional systems in high dimensions. HDVEC is designed to be general‑purpose, research‑grade, and production‑minded.

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

## Development Installation

- Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/Twistient/hdvec.git
cd hdvec

# Create a virtual environment (Python 3.10+)
uv venv -p 3.12

# Activate it
source .venv/bin/activate

# Install editable with dev
uv pip install -e ".[dev]"
```
- Optional extras:
  - Numba: `uv pip install -e ".[dev,numba]"`
  - Torch: `uv pip install -e ".[torch]"`
  - Docs: `uv pip install -e ".[docs]"`

## Quick Examples

**Binding and FPE**

```python
import numpy as np
from hdvec import bind, unbind
from hdvec.encoding.fpe import FPEEncoder, generate_base

D = 1024
base = generate_base(D)
encoder = FPEEncoder(D=D)

z1 = encoder(0.7)
z2 = encoder(1.3)
z_sum = bind(z1, z2)           # enc(0.7 + 1.3)
z1_rec = unbind(z_sum, z2)     # ≈ z1
```

**Grid/Field encoding (Scene)**

```python
import numpy as np
from hdvec.encoding.scene import FieldEncoder
from hdvec.encoding.positional import Positional2DTorus

D = 1024
positional = Positional2DTorus(D, beta=0.5)
rng = np.random.default_rng(0)
codebook = np.stack(
    [np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64) for _ in range(4)],
    axis=0,
)
values = (np.arange(16) % 4).reshape(4, 4)
encoder = FieldEncoder(positional=positional, value_codebook=codebook)
scene = encoder.encode_grid(values)
idx, score = encoder.read_cell(scene, 1, 2, values.shape)
scene_shifted = encoder.translate(scene, 1.0, 0.0)
```

**Residue HDC (RNS overlay)**

```python
from hdvec.encoding.residue import ResidueEncoder, res_pow_scalar, res_decode_int

moduli = [3, 5]
enc_res = ResidueEncoder(moduli=moduli, D=D)
vx = enc_res(7)
# multiply by integer k=2 via per‑modulus decode + rebundle
vx2 = res_pow_scalar(vx, 2, enc_res.bases)
# decode to integer via per‑modulus residues + CRT
x_rec = res_decode_int(vx2, enc_res.bases)
```

**GHRR**

```python
from hdvec.ghrr import sample_ghrr, gh_bind, gh_unbind, gh_project_unitary

G1 = sample_ghrr(8, 2)
G2 = sample_ghrr(8, 2)
G12 = gh_bind(G1, G2)
G1_rec = gh_unbind(G12, G2)
G12_u = gh_project_unitary(G12)
```

**Boolean Logic**

```python
from hdvec.encoding.boolean import BooleanEncoder, logic_and

encoder = BooleanEncoder(D=64)
a = encoder.encode(1)
b = encoder.encode(0)
result = logic_and(a, b, encoder)  # encodes 0
```

## Use Cases

- Structure‑aware perception and reasoning in high dimension
- Compositional encoding of sets, sequences, records, scenes
- Numeric/circular attributes (FPE) with translation‑equivariant ops (VFA)
- Large‑range integers and cycles (Residue HDC)
- Operator‑valued encodings and matrix‑like bindings (GHRR)

## Design & Performance

- NumPy‑first with optional Numba JIT (no‑op fallback); `complex64` default
- Recommended D: 8k–32k for concentration effects
- CI: Python 3.10/3.11/3.12; lint (ruff), format (black), types (mypy), tests (pytest)

## References

- Plate, T. (1995). *Holographic Reduced Representations (HRR)*
- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation*
- Frady, E. P. et al. (2021). *Computing on Functions Using Randomized Vector Representations*
- Kymn, C. J. et al. (2023). *Computing with Residue Numbers in High-Dimensional Representation*
- Yeung, C. et al. (2024). *Generalized Holographic Reduced Representations*
- Vergés, P. et al. (2025). *Learning Encoding Phasors with Fractional Power Encoding*

## Contributing

- Pre‑commit hooks: `pre-commit install` then `pre-commit run --all-files`
- Style: ruff + black; types: mypy; tests: pytest
- Please open issues/PRs with repro steps and concise diffs.

## Versioning & Releases

- Semantic versioning.
- Current version: 0.1.0 (alpha).
- See [CHANGELOG](CHANGELOG.md) for more details.

## License

LGPLv3 - see [LICENSE](LICENSE) file for details.
