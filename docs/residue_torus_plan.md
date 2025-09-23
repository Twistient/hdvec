# ResidueTorus — Coordinate→Residue Mapping Plan

Target: principled mapping from continuous (x, y) ∈ [0,1)² to residues per axis, with wrap and translation invariance. Outline:

1) Choose co‑prime moduli sets per axis (m_x, m'_x, …) to meet target range.
2) Map x to integer index n_x = floor(x * M_x) with M_x = ∏ m_xᵢ; compute residues n_x mod m_xᵢ; encode per‑modulus and bind across axes.
3) Provide readout inverse by residue decode + CRT per axis.

Add tests for wrap‑around and translation/readout invariance.

