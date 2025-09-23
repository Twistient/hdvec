"""Scene encoding cross-talk benchmark.

Measures round-trip accuracy and cleanup margin versus fill ratio using
HDVEC's positional FPE roles and FieldEncoder. This script vectorises
cell cleanup so it runs quickly even for larger grids.

Usage:
    python -m benchmarks.scene_crosstalk --dimension 8192 --size 10 --steps 21 --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hdvec.core import bind, inv
from hdvec.core.codebook import Codebook
from hdvec.encoding.positional import Positional2DTorus

try:  # optional plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None


def random_codebook(K: int, D: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(-np.pi, np.pi, size=(K, D))
    return np.exp(1j * theta).astype(np.complex64)


def encode_scene(
    grid: np.ndarray,
    positional: Positional2DTorus,
    codebook: np.ndarray,
    background: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scene_vector, position_grid)."""
    H, W = grid.shape
    pos_grid = positional.sample_grid(H, W)
    flat_pos = pos_grid.reshape(-1, positional.D)
    flat_vals = grid.reshape(-1)
    scene = (flat_pos * codebook[flat_vals]).sum(axis=0)
    return scene.astype(np.complex64), pos_grid


def decode_scene(
    scene: np.ndarray,
    pos_grid: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_indices, margins)."""
    flat_pos = pos_grid.reshape(-1, pos_grid.shape[-1])
    slots = flat_pos.conj() * scene  # broadcast
    cb = Codebook(codebook)
    idxs, scores = cb.topk_batch(slots, k=min(2, codebook.shape[0]))
    idxs = np.asarray(idxs)
    scores = np.asarray(scores)
    if scores.shape[1] == 1:
        margins = scores[:, 0]
    else:
        margins = scores[:, 0] - scores[:, 1]
    return idxs[:, 0].reshape(pos_grid.shape[:2]), margins.reshape(pos_grid.shape[:2])


def run_benchmark(
    D: int,
    size: int,
    steps: int,
    background: int,
    seed: int,
    plot: bool,
) -> None:
    rng = np.random.default_rng(seed)
    positional = Positional2DTorus(D=D, beta=0.5, rng=np.random.default_rng(seed + 1))
    codebook = random_codebook(K=10, D=D, rng=rng)

    ratios = np.linspace(0.0, 1.0, steps)
    accuracies: list[float] = []
    margins: list[float] = []

    pos_grid = positional.sample_grid(size, size)
    start = time.perf_counter()
    for ratio in ratios:
        grid = np.zeros((size, size), dtype=np.int32)
        num_active = int(ratio * grid.size)
        if num_active > 0:
            idx = rng.choice(grid.size, size=num_active, replace=False)
            grid.flat[idx] = rng.integers(1, codebook.shape[0], size=num_active)
        scene, _ = encode_scene(grid, positional, codebook, background)
        pred_grid, margin_grid = decode_scene(scene, pos_grid, codebook)
        accuracies.append(float((pred_grid == grid).mean()))
        margins.append(float(margin_grid.mean()))
    elapsed = time.perf_counter() - start

    print(f"dimension={D}, size={size}x{size}, steps={steps}, elapsed={elapsed:.2f}s")
    print("ratio\taccuracy\tavg_margin")
    for r, acc, m in zip(ratios, accuracies, margins):
        print(f"{r:.2f}\t{acc:.4f}\t{m:.4f}")

    if plot and plt is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(ratios, accuracies, marker="o")
        ax[0].set_title("Round-trip accuracy vs fill ratio")
        ax[0].set_xlabel("Fill ratio")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_ylim(0, 1.05)
        ax[0].grid(True)

        ax[1].plot(ratios, margins, marker="o", color="orange")
        ax[1].set_title("Average margin vs fill ratio")
        ax[1].set_xlabel("Fill ratio")
        ax[1].set_ylabel("Mean top-1 margin (cosine)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene encoding cross-talk benchmark")
    parser.add_argument("--dimension", type=int, default=8192, help="Vector dimension D")
    parser.add_argument("--size", type=int, default=10, help="Grid size (H=W)")
    parser.add_argument("--steps", type=int, default=21, help="Number of fill ratios to sweep")
    parser.add_argument("--background", type=int, default=0, help="Background color index")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots if available")
    args = parser.parse_args()

    run_benchmark(
        D=args.dimension,
        size=args.size,
        steps=args.steps,
        background=args.background,
        seed=args.seed,
        plot=args.plot,
    )
