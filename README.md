# hdvec

Hyperdimensional vector primitives and encoders (FPE, VFA, GHRR, RHC) with NumPy/Numba stubs.

Status: Alpha scaffolding with working implementations and tests. APIs and math may evolve.

## Installation (uv)

- Create a virtual environment (Python 3.10+):
  uv venv -p 3.12
- Activate it:
  source .venv/bin/activate
- Install in editable mode with dev extras:
  uv pip install -e ".[dev]"
- Optional extras:
  - Numba (CPU JIT acceleration): uv pip install -e ".[dev,numba]"
  - Torch: uv pip install -e ".[torch]"
  - Docs: uv pip install -e ".[docs]"

## Quickstart

```python
import numpy as np
from hdvec import bind, bundle, similarity, unbind, project_unitary
from hdvec.fpe import FPEEncoder, generate_base, encode_fpe
from hdvec.vfa import VFAEncoder

D = 1024
base = generate_base(D)
enc = FPEEncoder(D=D, dist="uniform")

z_r1 = enc(0.7)
z_r2 = enc(1.3)
z_sum = bind(z_r1, z_r2, op="hadamard")  # z(r1+r2)

points = np.array([0.2, 0.8])
alphas = np.array([1.0, -0.5])
vfa = VFAEncoder(base)
y_f = vfa.encode(points, alphas)
value_at_0_5 = vfa.readout(y_f, 0.5)
print(value_at_0_5)

# Grid/field encoding (see docs/quickstart_grid.md)
from hdvec.vfa import encode_grid, read_cell, translate_grid

K = 5
codebook = np.stack([
    np.exp(1j * np.random.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    for _ in range(K)
], axis=0)
values = (np.arange(16) % K).reshape(4, 4)
scene = encode_grid(values, [base, base], value_codebook=codebook)
idx, score = read_cell(scene, 1, 2, [base, base], value_codebook=codebook)
print("decoded index:", int(idx), "cosine:", float(score))

# Cleanup: batch nearest neighbors
from hdvec.core import topk_batch, nearest_batch
q = codebook.copy()  # pretend these are queries
idxs, scores = topk_batch(q, codebook, k=1)
print("top-1 indices:", idxs.ravel().tolist())
```

## Development

- Lint/format/type-check:
  - uv run ruff check .
  - uv run black --check .
  - uv run mypy .
- Tests: uv run pytest
- Slow tests (large-D): mark with @pytest.mark.slow; see docs/HDVEC_SCALING.md
- Pre-commit: pre-commit install
  - Configure hooks: pre-commit install
  - Run hooks on all files: pre-commit run --all-files

CI style checks
- Ruff runs on both `src/` and `tests/` in CI.
- Black enforces formatting across the repository (same as pre-commit).

## License

LGPL-3.0-or-later (see LICENSE)
