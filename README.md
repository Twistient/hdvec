# hdvec

Hyperdimensional vector primitives and encoders (FPE, VFA, GHRR, RHC) with NumPy/Numba stubs.

Status: Alpha scaffolding with minimal working implementations and tests. APIs and math may evolve.

## Installation (uv)

- Create a virtual environment (Python 3.10+):
  uv venv -p 3.12
- Activate it:
  source .venv/bin/activate
- Install in editable mode with dev extras:
  uv pip install -e ".[dev]"
- Optional extras:
  - Torch: uv pip install -e ".[torch]"
  - Numba (CPU JIT): uv pip install -e ".[numba]"  # enables optional JIT where available
  - Docs: uv pip install -e ".[docs]"

## Quickstart

```python
import numpy as np
from hdvec import bind, bundle, similarity, permute
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
```

## Development

- Lint/format/type-check:
  - uv run ruff check .
  - uv run black --check .
  - uv run mypy .
- Tests: uv run pytest
- Pre-commit: pre-commit install

## License

LGPL-3.0-or-later (see LICENSE)
