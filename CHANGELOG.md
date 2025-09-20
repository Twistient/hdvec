# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core: algebra ops (bind/unbind/bundle/similarity/permute), unitary projection, HRR conv/corr
- Encoders: FPE (scalar, circular/boolean, multi‑dim), VFA helpers, grid/field encoder
- Residue HDC: residue encode/add, per‑modulus decode + CRT, scalar multiply via decode+rebundle
- GHRR: sampling, bind/bundle/similarity, adjoint/unbind, projection
- Cleanup: Codebook utilities, batch top‑k/nearest
- Tooling: CI (ruff/black/mypy/pytest), docs scaffold, pre‑commit hooks

### Changed
- README rewritten with features, examples, and references

### Removed
- Internal planning/spec documents from version control (kept locally)

## [0.1.0] - YYYY-MM-DD
### Added
- Initial alpha tag

[Unreleased]: https://github.com/Twistient/hdvec/compare/0.1.0...HEAD
[0.1.0]: https://github.com/Twistient/hdvec/releases/tag/0.1.0
