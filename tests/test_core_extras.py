import numpy as np
from numpy.typing import NDArray
from typing import cast

from hdvec.core import bind, circ_conv, circ_corr, nearest, project_unitary, topk, unbind


def rand_phasor(dim: int, seed: int = 0) -> NDArray[np.complex64]:
    rng = np.random.default_rng(seed)
    return cast(
        NDArray[np.complex64],
        np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64),
    )


def test_unbind_recovers_operand():
    dim = 256
    a = rand_phasor(dim, 1)
    b = rand_phasor(dim, 2)
    z = bind(a, b)
    rec = unbind(z, b)
    # Similarity close to 1
    sim = (np.conj(a) * rec).sum().real / dim
    assert sim > 0.95


def test_project_unitary_sets_unit_modulus():
    dim = 128
    a = rand_phasor(dim) * 2.5
    pu = project_unitary(a)
    mags = np.abs(np.asarray(pu))
    assert np.allclose(mags, 1.0, atol=1e-5)


def test_topk_and_nearest_basic():
    dim, k = 64, 10
    rng = np.random.default_rng(0)
    codebook = np.stack(
        [np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64) for _ in range(k)],
        axis=0,
    )
    # Query near atom 3
    q = codebook[3] * np.exp(1j * 0.1).astype(np.complex64)
    idxs, scores = topk(q, codebook, k=3)
    assert idxs[0] == 3
    idx, score = nearest(q, codebook)
    assert idx == 3


def test_circ_conv_corr_inverse_property():
    dim = 128
    a = rand_phasor(dim, 3)
    b = rand_phasor(dim, 4)
    c = circ_conv(a, b)
    # Deconvolution in frequency domain should recover a (within tolerance)
    fa = np.fft.fft(np.asarray(a))
    fb = np.fft.fft(np.asarray(b))
    fc = np.fft.fft(np.asarray(c))
    a_rec = np.fft.ifft(fc / (fb + 1e-9))
    err = np.linalg.norm(np.asarray(a) - a_rec) / np.sqrt(dim)
    assert err < 1e-3
