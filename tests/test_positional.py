import numpy as np

from hdvec.encoding.positional import Positional2DTorus, ResidueTorus
from hdvec.encoding.scene import FieldEncoder


def test_positional_torus_translation():
    D = 64
    pos = Positional2DTorus(D, beta=0.5, rng=np.random.default_rng(0))
    grid = pos.sample_grid(4, 4)
    delta = pos.trans(0.25, 0.0)
    shifted = grid[0, 0] * delta
    target = pos.pos(0.25, 0.0)
    assert np.allclose(np.abs(shifted), np.abs(target), atol=1e-5)


def test_residue_torus_selection():
    torus = ResidueTorus(D=64, beta=0.5, moduli=[5, 7])
    code_a = torus.pos(0.1, 0.3)
    code_b = torus.pos(0.1, 0.3)
    assert np.allclose(code_a, code_b)


def test_field_encoder_roundtrip():
    positional = Positional2DTorus(D=64, beta=0.7, rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    codebook = np.stack(
        [np.exp(1j * rng.uniform(-np.pi, np.pi, size=64)).astype(np.complex64) for _ in range(4)],
        axis=0,
    )
    encoder = FieldEncoder(positional=positional, value_codebook=codebook)
    grid = np.array([[0, 1], [2, 3]], dtype=int)
    scene = encoder.encode_grid(grid)
    read = encoder.read_cell(scene, 1, 0, (2, 2))
    assert int(read[0]) == 2

    translated = encoder.translate(scene, 0.0, 0.5)
    assert translated.shape == scene.shape
