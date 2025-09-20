import numpy as np

from hdvec.permute import apply_perm, dihedral_permutations


def test_dihedral_rotations_cycle_to_identity():
    n = 5
    perms = dihedral_permutations(n)
    base = np.arange(n * n)
    rot90 = perms["rot90"]
    x = apply_perm(base, rot90)
    x = apply_perm(x, rot90)
    x = apply_perm(x, rot90)
    x = apply_perm(x, rot90)
    assert np.all(x == base)
