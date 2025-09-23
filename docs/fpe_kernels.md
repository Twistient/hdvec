# FPE Kernel Parameterization — Diagnostics

This note documents the effect of the phase distribution and bandwidth `beta` on the induced kernel. Use `hdvec.encoding.kernels.estimate_kernel` to empirically estimate similarity vs offset:

```python
import numpy as np
from hdvec.encoding.fpe import generate_base
from hdvec.encoding.kernels import estimate_kernel

D = 4096
base = generate_base(D, dist="gaussian", beta=0.5)
offsets = np.linspace(-3, 3, 41)
x, y = estimate_kernel(base, offsets)
```

Future work: add a small plotting helper to overlay the empirical kernel with the analytic expectation and include examples for `laplace`, `cauchy`, and `student` families.

