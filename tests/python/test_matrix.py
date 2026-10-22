"""Chord-length (system matrix) entry points of the Python API.

Numerical correctness of the chord computation is covered by the C++ suite
(tests/cpp/test_chord_reference.cpp). These tests check what the bindings add on top: array
conversion, the shape and dtype of the results, the triplet and flat-index conventions, and
that the dense, triplet and stored-sparse entry points agree with each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Image: 2 slices (y), 3 pixels in x (origin_x = 3 -> j = floor(3 - x)), 3 in z.
SHAPE = (2, 3, 3)
ORIGIN = (3.0, 0.0, 0.0)

# Two-point trajectories crossing zero to three voxel boundaries.
TRAJECTORIES = {
    "inside_one_voxel": ([1.2, 1.7], [0.5, 0.5], [0.2, 0.7]),
    "one_boundary_z": ([1.5, 1.5], [0.5, 0.5], [0.9, 1.4]),
    "two_boundaries_xz": ([1.1, 0.9], [0.5, 0.5], [0.85, 1.1]),
    "three_boundaries_xyz": ([1.1, 0.9], [0.95, 1.1], [0.9, 1.15]),
}


def dense(x, y, z, **kwargs) -> np.ndarray:
    from pct_tools.matrix import compute_matrix_elements

    return compute_matrix_elements(
        np.asarray(x, np.float32),
        np.asarray(y, np.float32),
        np.asarray(z, np.float32),
        SHAPE,
        ORIGIN,
        1.0,
        1.0,
        **kwargs,
    )


def test_compute_matrix_elements__shape_and_dtype() -> None:
    chords = dense(*TRAJECTORIES["inside_one_voxel"])

    assert chords.shape == SHAPE
    assert chords.dtype == np.float32


def test_compute_matrix_elements__single_voxel_chord() -> None:
    """A segment inside one voxel contributes its full length to that voxel and nothing else."""
    x, y, z = TRAJECTORIES["inside_one_voxel"]
    chords = dense(x, y, z)

    # i = floor(y), j = floor(3 - x), k = floor(z) for the first point (1.2, 0.5, 0.2).
    expected = np.zeros(SHAPE, dtype=np.float32)
    expected[0, 1, 0] = np.hypot(x[1] - x[0], z[1] - z[0])
    np.testing.assert_allclose(chords, expected, atol=1e-6)


def test_compute_matrix_elements__rejects_float64_and_lists() -> None:
    """Only float32 arrays are accepted; the conversion (a copy) is left to the caller."""
    from pct_tools.matrix import compute_matrix_elements

    x, y, z = TRAJECTORIES["two_boundaries_xz"]
    x32, y32 = np.asarray(x, np.float32), np.asarray(y, np.float32)

    with pytest.raises(TypeError, match="z must have dtype float32, not float64"):
        compute_matrix_elements(x32, y32, np.asarray(z, np.float64), SHAPE, ORIGIN, 1.0, 1.0)
    with pytest.raises(TypeError, match="z must be a numpy array of dtype float32, not list"):
        compute_matrix_elements(x32, y32, z, SHAPE, ORIGIN, 1.0, 1.0)


def test_compute_matrix_elements__n_threads_keyword() -> None:
    x, y, z = TRAJECTORIES["three_boundaries_xyz"]

    np.testing.assert_array_equal(dense(x, y, z, n_threads=2), dense(x, y, z, n_threads=1))


def test_compute_matrix_triplets__matches_dense() -> None:
    """Triplets are (trajectory index, flat voxel index, chord) with C-order flat indices."""
    from pct_tools.matrix import compute_matrix_triplets

    names = list(TRAJECTORIES)
    x, y, z = (np.array([TRAJECTORIES[n][axis] for n in names], dtype=np.float32) for axis in range(3))
    triplets = compute_matrix_triplets(x, y, z, SHAPE, ORIGIN, 1.0, 1.0)

    assert all(isinstance(t, tuple) and len(t) == 3 for t in triplets)
    assembled = np.zeros((len(names), np.prod(SHAPE)), dtype=np.float32)
    for row, col, value in triplets:
        assembled[row, col] += value
    for row, name in enumerate(names):
        np.testing.assert_allclose(assembled[row].reshape(SHAPE), dense(*TRAJECTORIES[name]), atol=1e-6)


@pytest.mark.parametrize("name", list(TRAJECTORIES))
def test_compute_and_store_sparse_matrix__round_trip_matches_dense(name: str, tmp_path: Path) -> None:
    from pct_tools.io import read_compressed_matrix
    from pct_tools.matrix import compute_and_store_sparse_matrix

    x, y, z = TRAJECTORIES[name]
    filename = str(tmp_path / "row.mat")
    compute_and_store_sparse_matrix(
        filename,
        np.asarray([x], np.float32),
        np.asarray([y], np.float32),
        np.asarray([z], np.float32),
        SHAPE,
        ORIGIN,
        1.0,
        1.0,
        n_threads=1,
        compression_level=1,
    )
    stored = read_compressed_matrix(filename)

    assert stored.shape == (1, np.prod(SHAPE))
    np.testing.assert_allclose(stored.toarray().reshape(SHAPE), dense(x, y, z), atol=1e-6)


# ---- Bindings read the coordinate arrays in place --------------------------------------
#
# The block entry points once took nested vectors, so pybind11 converted every sample through
# a Python object: about 2 s for a 20000 x 1025 block, forty times the chord computation.
# They now take NumPy buffers. The signature check guards the binding type; the layout check
# guards that every array layout is still accepted and gives the same result.


@pytest.mark.parametrize("function", ["compute_matrix_triplets", "compute_and_store_sparse_matrix"])
def test_block_bindings__take_numpy_buffers(function: str) -> None:
    import pct_tools._ext

    signature = getattr(pct_tools._ext, function).__doc__.splitlines()[0]
    for name in ("x", "y", "z"):
        assert f"{name}: numpy.ndarray[numpy.float32]" in signature, signature


def test_compute_matrix_triplets__array_layouts_agree() -> None:
    """float32 C-contiguous arrays are read in place; other float32 layouts are converted."""
    from pct_tools.matrix import compute_matrix_triplets

    names = list(TRAJECTORIES)
    x, y, z = (np.array([TRAJECTORIES[n][axis] for n in names], dtype=np.float32) for axis in range(3))
    expected = compute_matrix_triplets(x, y, z, SHAPE, ORIGIN, 1.0, 1.0)

    layouts = {
        "fortran": [np.asfortranarray(a) for a in (x, y, z)],
        "strided": [np.repeat(a, 2, axis=1)[:, ::2] for a in (x, y, z)],
    }
    for layout, (xx, yy, zz) in layouts.items():
        assert compute_matrix_triplets(xx, yy, zz, SHAPE, ORIGIN, 1.0, 1.0) == expected, layout


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        (((4, 2), (4, 2), (4,)), "2 dimensions"),
        (((4, 2), (3, 2), (4, 2)), "same number of trajectories"),
        (((4, 2), (4, 2), (4, 3)), "same size"),
        (((0, 2), (0, 2), (0, 2)), "at least one trajectory"),
        (((4, 0), (4, 0), (4, 0)), "2 dimensions"),
    ],
)
def test_compute_matrix_triplets__rejects_mismatched_arrays(shapes, message: str) -> None:
    from pct_tools.matrix import compute_matrix_triplets

    x, y, z = (np.zeros(shape, dtype=np.float32) for shape in shapes)
    with pytest.raises(ValueError, match=message):
        compute_matrix_triplets(x, y, z, SHAPE, ORIGIN, 1.0, 1.0)


def test_sampling_step_crossing_two_boundaries_raises() -> None:
    """A segment crossing two z boundaries in one step is rejected by every entry point."""
    from pct_tools.matrix import compute_matrix_triplets

    x, y, z = [1.5, 1.5], [0.5, 0.5], [0.3, 2.6]
    with pytest.raises(ValueError, match="sampling step too coarse"):
        dense(x, y, z)
    with pytest.raises(ValueError, match="sampling step too coarse"):
        compute_matrix_triplets(
            np.asarray([x], np.float32),
            np.asarray([y], np.float32),
            np.asarray([z], np.float32),
            SHAPE,
            ORIGIN,
            1.0,
            1.0,
        )
