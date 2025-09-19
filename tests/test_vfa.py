import numpy as np

from hdvec.fpe import generate_base
from hdvec.vfa import VFAEncoder, encode_function, readout


def test_encode_and_readout_runs():
    D = 128
    base = generate_base(D)
    points = np.array([0.2, 0.8])
    alphas = np.array([1.0, -0.5])
    y_f = encode_function(points, alphas, base)
    enc = VFAEncoder(base)
    val = enc.readout(y_f, 0.5)
    assert isinstance(val, float)
