import numpy as np
import pytest
from numpy.typing import NDArray
from typing import cast

from hdvec.config import get_config
from hdvec.core import (
    bind,
    circ_conv,
    circ_corr,
    nearest,
    nearest_batch,
    project_unitary,
    topk,
    topk_batch,
    unbind,
)
from hdvec.errors import ConfigurationError, InvalidBindingError


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


def test_topk_batch_and_nearest_batch():
    dim, k = 32, 5
    rng = np.random.default_rng(0)
    C = np.stack(
        [np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64) for _ in range(k)],
        axis=0,
    )
    Q = C.copy()  # queries identical to codebook atoms
    idxs, scores = topk_batch(Q, C, k=1)
    assert np.all(idxs.ravel() == np.arange(k))
    idx1, sc1 = nearest_batch(Q, C)
    assert np.all(idx1 == np.arange(k))


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


def test_lcc_binding_matches_manual_fft():
    cfg = get_config()
    old_blocks = cfg.lcc_blocks
    cfg.lcc_blocks = 4
    try:
        dim = 64
        rng = np.random.default_rng(0)
        a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64)
        b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64)
        result = np.asarray(bind(a, b, op="lcc"))
        block_len = dim // cfg.lcc_blocks
        fa = np.fft.fft(a.reshape(cfg.lcc_blocks, block_len), axis=-1)
        fb = np.fft.fft(b.reshape(cfg.lcc_blocks, block_len), axis=-1)
        expected = np.fft.ifft(fa * fb, axis=-1).reshape(dim).astype(np.complex64)
        assert np.allclose(result, expected, atol=1e-6)
    finally:
        cfg.lcc_blocks = old_blocks


def test_lcc_binding_batched():
    cfg = get_config()
    old_blocks = cfg.lcc_blocks
    cfg.lcc_blocks = 2
    try:
        batch, dim = 3, 32
        rng = np.random.default_rng(1)
        a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(batch, dim))).astype(np.complex64)
        b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(batch, dim))).astype(np.complex64)
        result = np.asarray(bind(a, b, op="lcc"))
        block_len = dim // cfg.lcc_blocks
        fa = np.fft.fft(a.reshape(batch, cfg.lcc_blocks, block_len), axis=-1)
        fb = np.fft.fft(b.reshape(batch, cfg.lcc_blocks, block_len), axis=-1)
        expected = np.fft.ifft(fa * fb, axis=-1).reshape(batch, dim).astype(np.complex64)
        assert np.allclose(result, expected, atol=1e-6)
    finally:
        cfg.lcc_blocks = old_blocks


def test_lcc_binding_requires_config():
    cfg = get_config()
    old_blocks = cfg.lcc_blocks
    cfg.lcc_blocks = None
    try:
        a = np.ones(16, dtype=np.complex64)
        b = np.ones(16, dtype=np.complex64)
        with pytest.raises(ConfigurationError):
            bind(a, b, op="lcc")
    finally:
        cfg.lcc_blocks = old_blocks


def test_bind_invalid_operator():
    a = np.ones(8)
    b = np.ones(8)
    with pytest.raises(InvalidBindingError):
        bind(a, b, op="unknown")
