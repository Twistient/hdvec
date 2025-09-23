import numpy as np

from hdvec.permute import dihedral_permutations, apply_perm


def test_d4_group_relations():
    n = 4
    perms = dihedral_permutations(n)
    base = np.arange(n * n)

    # Rotations compose: rot90^4 = identity
    x = base.copy()
    for _ in range(4):
        x = apply_perm(x, perms["rot90"])
    assert np.all(x == base)

    # Reflections are involutions: flipx^2 = identity, flipy^2 = identity
    for name in ["flipx", "flipy", "flipdiag", "flipanti"]:
        y = apply_perm(base, perms[name])
        y = apply_perm(y, perms[name])
        assert np.all(y == base)
