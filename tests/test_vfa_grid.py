import numpy as np
from hdvec.encoding.fpe import generate_base
from hdvec.encoding.vfa import encode_grid, read_cell, translate_grid


def rand_codebook(k: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)).astype(np.complex64) for _ in range(k)],
        axis=0,
    )


def test_encode_grid_and_read_cell_roundtrip():
    dim = 128
    h_, w_ = 4, 4
    k = 5
    base_u = generate_base(dim, rng=np.random.default_rng(0))
    base_v = generate_base(dim, rng=np.random.default_rng(1))
    pos_bases = [base_u, base_v]
    codebook = rand_codebook(k, dim, seed=2)
    values = np.arange(h_ * w_, dtype=int).reshape(h_, w_) % k
    scene = encode_grid(values, pos_bases, value_codebook=codebook)
    # Probe a couple of cells
    for i, j in [(0, 0), (2, 3)]:
        idx_score = read_cell(scene, i, j, pos_bases, value_codebook=codebook)
        idx = int(idx_score[0])
        assert idx == int(values[i, j])


def test_translate_grid_integer_shift():
    dim = 128
    h_, w_ = 4, 4
    k = 3
    base_u = generate_base(dim, rng=np.random.default_rng(3))
    base_v = generate_base(dim, rng=np.random.default_rng(4))
    pos_bases = [base_u, base_v]
    codebook = rand_codebook(k, dim, seed=5)
    values = (np.arange(h_ * w_, dtype=int).reshape(h_, w_) * 2) % k
    scene = encode_grid(values, pos_bases, value_codebook=codebook)
    dx, dy = 1.0, 0.0
    scene_shift = translate_grid(scene, dx, dy, pos_bases)
    # A cell (i,j) should move to (i+1, j) modulo bounds in our simple encoding
    # (since we don't wrap here, just check readout consistency after inverse move)
    # Read at (i+1,j) and compare with original (i,j)
    i, j = 1, 2
    idx_orig = int(read_cell(scene, i, j, pos_bases, value_codebook=codebook)[0])
    idx_shifted_back = int(read_cell(scene_shift, i + 1, j, pos_bases, value_codebook=codebook)[0])
    assert idx_orig == idx_shifted_back
