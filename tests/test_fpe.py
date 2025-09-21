import numpy as np

from hdvec.core import bind
from hdvec.encoding.fpe import (
    FPEEncoder,
    convolve,
    encode_fpe,
    generate_base,
    probe,
    readout,
    shift,
)


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


def test_vfa_readout_shift_convolution():
    D = 128
    base = generate_base(D)
    # build simple function y_f = z(a) + 0.5*z(b)
    a, b = 0.2, 0.8
    y_f = encode_fpe(a, base) + 0.5 * encode_fpe(b, base)

    # readout should recover approximate values at anchors
    val_a = readout(y_f, a, base)
    val_b = readout(y_f, b, base)
    assert val_a > val_b

    # shift by delta and check correlation
    delta = 0.1
    y_shift = shift(y_f, delta, base)
    expected = encode_fpe(a + delta, base) + 0.5 * encode_fpe(b + delta, base)
    assert np.allclose(np.asarray(y_shift), expected, atol=1e-5)

    # Convolution should correspond to binding
    kernel = encode_fpe(0.5, base)
    conv = convolve(y_f, kernel)
    assert np.allclose(conv, y_f * kernel)

    # Probe multiple points
    xs = np.array([a, b])
    vals = probe(y_f, xs, base)
    assert vals.shape == (2,)
