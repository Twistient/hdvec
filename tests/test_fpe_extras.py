import numpy as np

from hdvec.fpe import (
    encode_boolean,
    encode_circular,
    encode_fpe,
    encode_fpe_vec,
    generate_base,
    make_circular_base,
)


def test_fpe_vec_matches_separable_binding():
    D = 128
    rng = np.random.default_rng(0)
    b1 = generate_base(D, rng=rng)
    b2 = generate_base(D, rng=rng)
    x = np.array([0.7, -1.3], dtype=float)
    v = encode_fpe_vec(x, [b1, b2])
    v_sep = encode_fpe(float(x[0]), b1) * encode_fpe(float(x[1]), b2)
    assert np.allclose(v, v_sep, atol=1e-6)


def test_circular_periodicity():
    D = 64
    L = 8
    baseL = make_circular_base(D, L, rng=np.random.default_rng(1))
    r = 3.4
    v1 = encode_circular(r, L, baseL)
    v2 = encode_circular(r + L, L, baseL)
    assert np.allclose(v1, v2, atol=1e-6)


def test_boolean_L2_encoding():
    D = 64
    base2 = make_circular_base(D, 2, rng=np.random.default_rng(2))
    z0 = encode_boolean(0, base2)
    z1 = encode_boolean(1, base2)
    assert np.allclose(z0, np.ones(D, dtype=np.complex64))
    assert np.allclose(z1, base2)
