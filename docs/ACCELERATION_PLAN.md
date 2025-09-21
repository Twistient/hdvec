# Acceleration Plan — Torch Backend and Numba/CuPy Tracks

Scope:
- Torch backend: lazy tensor conversion for bind/permute/top‑k; optional CUDA FFT for CC/LCC.
- Numba/CuPy: accelerate hot loops (block FFTs, batched similarities) behind capability flags.

Tasks:
1) Backend registry: add `torch` Backend with capability detection; tests with skip markers.
2) Implement `bind_hadamard_torch`, `permute_torch`, `topk_torch` with dtype parity.
3) Optional: `cc_torch` via torch.fft; ensure unitary/Hermitian constraints.
4) Numba‑accelerate similarity and lcc block transforms; CuPy parity under `cupy` extra.

Risks:
- Device placement/gradient semantics (Torch): remain explicit, no autograd exposed by default.
- CI coverage: optional workflow with matrix for extras to avoid CI flakiness.

