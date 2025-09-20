# HDVEC Scaling & Validation Notes

This note summarizes practical guidance on dimensionality `D`, statistical properties to expect as `D` grows, and an outline of “slow” tests to validate concentration‑of‑measure and kernel behavior. It complements the API spec in `docs/HDVEC_SPEC.md`.

## Dimensionality (D)

- Recommended ranges for production: `D ∈ {8192 … 32768}` (per HOLOGRAM V1/V2).
- Unit tests can use small `D` (64–1024) to verify algebraic identities (bind/unbind, FFT deconvolution, periodicity) quickly; these do not rely on concentration.
- Phenomena that improve with large `D`:
  - Pseudo‑orthogonality of random atoms (cosine concentrated near 0; variance ~ O(1/D)).
  - Cleanup performance and bundling capacity (SNR ~ O(√(D/m)) for m items bundled, domain‑dependent).
  - FPE kernel concentration and readout stability for function representations.

## Random atoms & orthogonality

- For i.i.d. random phasor components, pairwise inner products have mean 0 and variance ~ 1/D.
- As `D` increases, the distribution tightens around 0; the maximum spurious similarity across N candidates scales like √(log N / D) (extreme‑value heuristic).

## Bundling capacity (back‑of‑envelope)

- If `m` independent atoms are bundled, individual constituents remain retrievable via cleanup so long as the constituent’s signal dominates the accumulated cross‑terms/noise. A practical rule of thumb in phasor spaces is `m = O(D)` for tolerant cleanup, with constants depending on thresholds and codebook sizes. Validate empirically for your task.

## FPE kernels (VFA view)

- Let base phases φ be drawn i.i.d. from a distribution with characteristic function ϕ(t) = E[e^{i t φ}]. Then the induced translation‑invariant kernel is

  K(Δ) = E[e^{i φ Δ}] = ϕ(Δ).

- Examples:
  - Uniform φ ∼ U[−π, π] → `K(Δ) = sin(πΔ)/(πΔ)` (sinc‑like decay).
  - Cauchy φ → characteristic function `e^{-|Δ|}` → exponential‑like decay (heavier tails).
- Implication: base phase distribution shapes similarity falloff in the induced RKHS. Use this to tune bandwidths for readout/shift robustness.

## Slow test plan (opt‑in)

Mark these as `@pytest.mark.slow` and run periodically or in nightly CI.

1) Orthogonality concentration
- For D ∈ {2k, 8k, 32k}, sample many random atom pairs; verify std(cosine) ~ c/√D and max cosine decreases with D.

2) FPE kernel convergence
- Fix base distribution (uniform vs cauchy). For a grid of Δ values, estimate empirical ⟨v(x), v(x+Δ)⟩ and compare against the analytic kernel K(Δ). Tighten tolerance as D increases.

3) Cleanup ROC vs bundle size
- Bundle `m` random atoms at various `m` and measure retrieval hit‑rate vs codebook size and `D`. Plot curves, pick operating points for HOLOGRAM.

4) Unbinding under load
- Superpose `t` bound pairs, then unbind one role and cleanup; measure accuracy vs `t` and `D`.

## Performance notes

- Hadamard binding and similarity are O(D); FFT (HRR conv/corr) is O(D log D).
- Optional Numba JIT can accelerate hot scalar loops; HDVEC uses a no‑op fallback when Numba isn’t available.
- Keep dtype `complex64` to halve bandwidth vs `complex128`.

## How to run slow tests

```
pytest -m slow  # after marking slow tests
```

## Defaults & configuration

- Library default `D=1024` is intended for development. For HOLOGRAM tasks that rely on concentration, set `hdvec.config.get_config().D` to a larger value (e.g., 8192–32768) or pass explicit bases of that size to encoders.

