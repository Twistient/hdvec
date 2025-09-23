# Decoding & Resonators

`hdvec.decoding` provides only stateless helpers that mirror the reference spec:

- `AnchorMemory` implements a minimal clean-up memory that stores anchors in an
  HD codebook and retrieves the nearest neighbour by cosine similarity.
- `decode_point` performs direct template matching against a codebook of anchor
  vectors and returns the winning index and margin.

Iterative resonators, residue-specific demixers, and other dynamical decoders
live in the downstream [`hologram-resonator`](https://github.com/Twistient/hologram-resonator)
package. That project wraps the stateless utilities exposed here (`residue_correlations`
and `residue_initial_guess`) and adds annealing loops, step-size scheduling, and
application-specific stopping criteria.

If you need iterative demixing, import the resonator package and feed it the
stateless correlations/initial guesses from HDVec. Keeping the dynamics outside
this repository preserves the contract boundary between HDVec building blocks and
HOLOGRAM runtime components.
