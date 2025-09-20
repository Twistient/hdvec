# Quickstart: Grid/Field Encoding with VFA + FPE

HDVEC lets you encode grids (e.g., ARC panels) into a single hypervector using FPE for positions and a value codebook for per-cell content. You can then read cells and translate scenes in pure HD space.

## Encode a grid

```python
import numpy as np
from hdvec.fpe import generate_base
from hdvec.vfa import encode_grid, read_cell, translate_grid

# Vector dimension
D = 1024

# Positional bases for (u, v)
base_u = generate_base(D)
base_v = generate_base(D)
pos_bases = [base_u, base_v]

# Codebook of K values/colors (here random phasors)
K = 5
rng = np.random.default_rng(0)
codebook = np.stack([
    np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    for _ in range(K)
], axis=0)

# Grid of integer values in [0..K)
H, W = 4, 4
values = (np.arange(H*W) % K).reshape(H, W)

# Encode to a single scene hypervector
scene = encode_grid(values, pos_bases, value_codebook=codebook)

# Read back a cell (i, j)
i, j = 2, 3
idx, score = read_cell(scene, i, j, pos_bases, value_codebook=codebook)
print("decoded index:", int(idx), "cosine:", float(score))

# Translate by (+1, 0)
scene_shift = translate_grid(scene, 1.0, 0.0, pos_bases)
```

Notes
- Position codes use separable FPE bases; translation is a Hadamard bind with the translation code.
- You can substitute learned codebooks or FPE‑encoded continuous values for richer scenes.

