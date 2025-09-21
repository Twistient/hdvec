import numpy as np

from hdvec.encoding.residue import (
    ResidueBases,
    encode_residue,
    residue_initial_guess,
)


def test_residue_initial_guess_matches_true_residues():
    moduli = [3, 5]
    D = 512
    bases = ResidueBases.from_moduli(moduli, D=D, rng=np.random.default_rng(0))
    for x in [0, 1, 2, 7, 14, 23]:
        v = encode_residue(x, bases)
        guess = residue_initial_guess(v, bases)
        true = [x % m for m in moduli]
        assert guess == true

