# HDVEC Engineering Task List (V1 → V1.x)

This checklist captures the remaining engineering work to take HDVEC from the current scaffold to a robust, consumable library for the HOLOGRAM buildout while remaining generally useful.

## Ready for implementation (V1)

- Core API polish
  - [ ] Add docstring examples for: `topk/nearest`, `topk_batch/nearest_batch`, `project_unitary`, `circ_conv/circ_corr`, `Codebook` methods.
  - [ ] Batch variants for bind/bundle (broadcast-friendly over `(N,D)`), plus simple `(K,D) vs (N,D)` convenience wrappers.
  - [ ] Ensure all public functions accept `np.ndarray | BaseVector` consistently and return `Vec`/`np.ndarray` per spec; finalize return types in docstrings.

- FPE/VFA
  - [ ] Add docstring examples for `encode_fpe_vec`, `make_circular_base`, `encode_circular`, `encode_boolean` (periodic / boolean usage).
  - [ ] Vectorize `encode_grid` internals (remove Python loops for large grids), while keeping a readable implementation.
  - [ ] Add optional value-encoder hook (callable) to `encode_grid` to support learned/continuous values alongside codebooks.

- GHRR (V1.x-ready stub but extend in V1)
  - [ ] Add `gh_unbind` (per-slice adjoint) and `gh_project_unitary` (per-slice QR projection) helpers.
  - [ ] Add small tests for `gh_bundle` projection invariants and adjoint round-trips.

- Batch cleanup & utilities
  - [ ] Expose `Codebook.topk_batch/nearest_batch` convenience delegating to core batch functions.
  - [ ] Micro-benchmarks for batch `topk/nearest` and single-query variants at `D ∈ {8k, 16k, 32k}`.

- Docs
  - [ ] Expand function docstrings with inline examples and shape notes (core, fpe, vfa, permute, ghrr).
  - [ ] Add “API reference” pages for residue and ghrr with examples.
  - [ ] Quickstart: add a small FPE/VFA demo plotting kernel similarity decay (link to Scaling doc).

- Tests
  - [ ] Add pytest markers and first “slow” large-D tests:
    - [ ] Orthogonality concentration: std/cosine ~ O(1/√D) and max-cosine trend.
    - [ ] FPE kernel convergence for uniform/cauchy bases (empirical ⟨v(x),v(x+Δ)⟩ vs analytic K(Δ)).
    - [ ] Cleanup ROC vs bundle size (report curves, thresholds documented).
    - [ ] Unbinding under load (superposition of t bound pairs): recovery vs t.

- CI
  - [ ] Nightly workflow job to run `-m slow` on a single Python version and report artifacts (plots/tables as attachments or summary).
  - [ ] Optional Numba job matrix (import check + small perf sanity) with `numba` extra.

## Post‑V1 Roadmap (V1.x)

- GHRR expansion
  - [ ] Structured operators: block-circulant/unitary families; parameterized sampling.
  - [ ] Interop with `Vec` (e.g., apply per-slice operator to vector views) for experimental operator semantics.

- Residue HDC (held per request)
  - [ ] Dedicated multiplicative/exponentiation-binding operator surface.
  - [ ] Resonator/FOAM demixing (likely sibling library, but with clean hooks).
  - [ ] Hexagonal residue lattices for 2D coordinates.

- Performance
  - [ ] Batched bind/bundle on GPU backends (Torch/JAX adapters, if/when enabled).
  - [ ] FFT backend configurability (e.g., pocketfft, pyFFTW) behind a simple switch.

- Packaging & release
  - [ ] Sphinx site build & publish (gh-pages or ReadTheDocs).
  - [ ] Versioning policy & changelog; 0.1.0 alpha → 0.2.0 once slow tests & docs are in.
  - [ ] Optional: publish to PyPI when HOLOGRAM downstream interface stabilizes.

## Nice-to-haves

- Bench suite (scripts/bench.py) to compare ops/sec for bind/bundle/topk across D.
- Example notebooks: FPE kernel shapes; grid encoding/translation; batch cleanup demo.

---

Use this checklist as the running backlog. I can start by vectorizing `encode_grid`, adding docstring examples, and implementing the first two slow tests if you want those prioritized.

