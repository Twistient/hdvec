import numpy as np

from hdvec.fpe import FPEEncoder, generate_base, encode_fpe
from hdvec.core import bind


def test_generate_base_unitary():
    base = generate_base(128)
    assert base.shape == (128,)
    assert np.allclose(np.abs(base), 1.0, atol=1e-6)


def test_fpe_additivity():
    D = 256
    base = generate_base(D)
    z1 = encode_fpe(0.7, base)
    z2 = encode_fpe(1.3, base)
    z12 = encode_fpe(0.7 + 1.3, base)
    assert np.allclose(bind(z1, z2, op="hadamard"), z12, atol=1e-6)


def test_encoder_call():
    enc = FPEEncoder(D=64)
    z = enc(1.5)
    assert z.shape == (64,)
