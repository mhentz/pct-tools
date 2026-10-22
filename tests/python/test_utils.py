"""Matrix utilities: recompression and subsampling of stored matrix/vector pairs."""

from pathlib import Path

import numpy as np
import pytest

ROWS = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
COLS = np.array([0, 2, 1, 0, 3, 3], dtype=np.int32)
VALUES = np.arange(1, 7, dtype=np.float32)
SHAPE = (4, 4)


def write_matrix(filename: Path) -> None:
    from pct_tools.io import construct_matrix

    construct_matrix(str(filename), ROWS, COLS, VALUES, shape=SHAPE)


def test_recompress_matrix__round_trip(tmp_path: Path) -> None:
    from pct_tools.io import read_compressed_matrix
    from pct_tools.utils import recompress_matrix

    original = tmp_path / "a.mat"
    recompressed = tmp_path / "b.mat"
    write_matrix(original)
    recompress_matrix(str(original), str(recompressed), compression_level=1, codec="zstd")

    np.testing.assert_array_equal(
        read_compressed_matrix(str(recompressed)).toarray(),
        read_compressed_matrix(str(original)).toarray(),
    )


@pytest.fixture
def pair_tree(tmp_path: Path) -> Path:
    """The ``<root>/matrix-<pw>mm/<projection>.mat`` and ``<root>/b/b_<projection>.vec`` layout."""
    from pct_tools.io import store_vector

    root = tmp_path / "in"
    (root / "matrix-1.0mm").mkdir(parents=True)
    (root / "b").mkdir()
    write_matrix(root / "matrix-1.0mm" / "90.mat")
    store_vector(np.arange(SHAPE[0], dtype=np.float32), str(root / "b" / "b_90.vec"))
    return root


def test_subsample_matrix_vector_pair__keeps_matching_rows(pair_tree: Path, tmp_path: Path) -> None:
    from pct_tools.io import read_compressed_matrix, read_vector
    from pct_tools.utils import subsample_matrix_vector_pair

    out = tmp_path / "out"
    (out / "matrix-1.0mm").mkdir(parents=True)  # the output tree is not created automatically
    (out / "b").mkdir()
    subsample_matrix_vector_pair(str(pair_tree), "90", str(out), 0.5, 1.0, seed=1)

    matrix = read_compressed_matrix(str(out / "matrix-1.0mm" / "90.mat")).toarray()
    vector = read_vector(str(out / "b" / "b_90.vec"))
    full = read_compressed_matrix(str(pair_tree / "matrix-1.0mm" / "90.mat")).toarray()

    assert matrix.shape == (2, SHAPE[1])
    assert vector.shape == (2,)
    # Each kept row is the original row the b entry points to.
    for row, b in zip(matrix, vector, strict=True):
        np.testing.assert_array_equal(row, full[int(b)])


def test_subsample_matrix_vector_pair__negative_seed_raises(pair_tree: Path, tmp_path: Path) -> None:
    from pct_tools.utils import subsample_matrix_vector_pair

    with pytest.raises(ValueError, match="non-negative"):
        subsample_matrix_vector_pair(str(pair_tree), "90", str(tmp_path / "out"), 0.5, 1.0, seed=-1)


def test_subsample_matrix_vector_pair__ratio_out_of_range_raises(pair_tree: Path, tmp_path: Path) -> None:
    from pct_tools.utils import subsample_matrix_vector_pair

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        subsample_matrix_vector_pair(str(pair_tree), "90", str(tmp_path / "out"), 1.5, 1.0)
