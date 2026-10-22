"""Time the system-matrix construction for several sampling steps and thread counts.

Builds a block of trajectories through a 16 x 256 x 256 image of 1 mm voxels and times two
entry points for several sampling steps and thread counts:

* ``triplets``: :func:`pct_tools.matrix.compute_matrix_triplets`, which also converts the
  result into a Python list of tuples.
* ``store``: :func:`pct_tools.matrix.compute_and_store_sparse_matrix` to a temporary file
  with Blosc compression level 1. This is the production path: chords, sparse assembly,
  compression and writing, all on the C++ side.

The coordinate arrays are read in place by the binding, so ``store`` measures the C++ work
itself. ``triplets`` additionally builds a Python list of one tuple per non-zero entry, which
costs more than the chords for large blocks; use ``store`` for production and ``triplets``
when the entries are needed in memory.

Two geometries are used: ``gentle`` trajectories run nearly along z with a slight curvature,
``diagonal`` ones cross a voxel boundary roughly every millimetre along x and z.

Chord lengths are exact for the sampled polyline as long as no segment crosses more than one
voxel boundary along an axis, so for straight trajectories the result is the same at every
step up to one voxel and, for the gently curved ones used here, agrees to the precision
printed. Steps that cross two boundaries along an axis raise ``ValueError``; the default
steps stay within the limit for both geometries.

Run with ``python examples/benchmark_chords.py [--trajectories N] [--repeats R]``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from pct_tools.matrix import compute_and_store_sparse_matrix, compute_matrix_triplets

IMG_SHAPE = (16, 256, 256)
ORIGIN = (256.0, 0.0, 0.0)  # voxel (0, 0, 0) corner; x decreases with the j index
DEPTH = 256.0


def trajectories(n: int, step: float, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, z)`` arrays of shape ``(n, n_points)`` for the given geometry."""
    b = np.arange(n)[:, None]
    z = np.arange(0.0, DEPTH + step / 2, step, dtype=np.float64)[None, :]

    if mode == "gentle":
        x0 = 8.0 + 240.0 * (b % 97) / 96.0
        slope_x = 0.02 * ((b % 7) - 3)
        slope_y = 0.0
    elif mode == "diagonal":
        x0 = 128.0 + 100.0 * ((b % 97) - 48) / 48.0
        slope_x = np.where(b % 2 == 1, 1.0, -1.0)
        slope_y = 0.1 * ((b % 3) - 1)
    else:
        message = f"unknown mode {mode!r}"
        raise ValueError(message)

    y0 = 1.0 + 14.0 * (b % 13) / 12.0
    x = x0 + slope_x * z + 0.3 * np.sin(0.05 * z)
    y = y0 + slope_y * z + 0.2 * np.sin(0.03 * z + b)
    z = np.broadcast_to(z, x.shape)
    x, y, z = (np.ascontiguousarray(a, dtype=np.float32) for a in (x, y, z))
    return x, y, z


def time_block(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_threads: int,
    repeats: int,
    filename: str,
) -> tuple[float, float, int, float]:
    """Best wall times of the two entry points, number of triplets and total chord length."""
    best_triplets = best_store = float("inf")
    triplets: list[tuple[int, int, float]] = []
    compute_matrix_triplets(x, y, z, IMG_SHAPE, ORIGIN, 1.0, 1.0, n_threads=n_threads)  # warm-up
    for _ in range(repeats):
        start = time.perf_counter()
        triplets = compute_matrix_triplets(x, y, z, IMG_SHAPE, ORIGIN, 1.0, 1.0, n_threads=n_threads)
        best_triplets = min(best_triplets, time.perf_counter() - start)

        start = time.perf_counter()
        compute_and_store_sparse_matrix(
            filename,
            x,
            y,
            z,
            IMG_SHAPE,
            ORIGIN,
            1.0,
            1.0,
            n_threads=n_threads,
            compression_level=1,
        )
        best_store = min(best_store, time.perf_counter() - start)
    return best_triplets, best_store, len(triplets), float(sum(value for _, _, value in triplets))


def main() -> None:
    """Parse the command line, run the benchmark and print one table per entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trajectories", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()

    print(
        f"{'mode':9} {'step/mm':>8} {'threads':>7} {'points':>7} {'triplets/ms':>11} {'store/ms':>9}"
        f" {'triplets':>9} {'sum chords':>12}",
    )
    with tempfile.TemporaryDirectory() as tmp:
        filename = str(Path(tmp) / "block.mat")
        for mode in ("gentle", "diagonal"):
            for step in args.steps:
                x, y, z = trajectories(args.trajectories, step, mode)
                for n_threads in args.threads:
                    t_triplets, t_store, n_triplets, total = time_block(
                        x,
                        y,
                        z,
                        n_threads,
                        args.repeats,
                        filename,
                    )
                    print(
                        f"{mode:9} {step:8.2f} {n_threads:7d} {x.shape[1]:7d} {1e3 * t_triplets:11.2f}"
                        f" {1e3 * t_store:9.2f} {n_triplets:9d} {total:12.1f}",
                    )


if __name__ == "__main__":
    main()
