# Localized Circular Convolution (LCC) — Demo

This note sketches block‑local circular convolution (LCC) usage in HDVEC. LCC splits the trailing dimension into `blocks` and performs circular convolution within each block.

Example (Python):

```python
import numpy as np
from hdvec.core import bind
from hdvec.config import get_config

cfg = get_config()
cfg.lcc_blocks = 4
D = 64
rng = np.random.default_rng(0)
a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
z = bind(a, b, op="lcc")
```

Future work: expose LCC‑specific base constructors that yield block‑sparse encodings for integer arguments (per VFA §2), and add a test validating block sparsity.

