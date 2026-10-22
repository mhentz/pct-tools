"""Round trips through the compressed matrix format via the Python wrappers.

Checks that construct_matrix and read_compressed_matrix agree on shape and values, including
the shuffle method and compression level arguments and matrices with more than one row.
"""

from pathlib import Path


def test_construct_matrix__one_row_all_ones(tmp_path: Path):
    import numpy as np

    from pct_tools.io import ShuffleMethod, construct_matrix, read_compressed_matrix

    out_dir = tmp_path / "matrices"
    out_dir.mkdir()

    test_matrix_path = out_dir / "test.mat"

    construct_matrix(
        filename=str(test_matrix_path),
        row_indexes=np.array([0, 0, 0, 0], dtype=np.int32),
        col_indexes=np.array([0, 1, 2, 3], dtype=np.int32),
        values=np.array([1, 1, 1, 1], dtype=np.float32),
        shape=(1, 4),
        compression_level=1,
        shuffle_method=ShuffleMethod.BLOSC_BITSHUFFLE,
        verbose_level=0,
    )
    actual_array = read_compressed_matrix(str(test_matrix_path), 0).toarray()

    expected_array = np.array([[1, 1, 1, 1]], dtype=np.float32)

    np.testing.assert_equal(actual_array, expected_array)


def test_construct_matrix__two_rows(tmp_path: Path):
    import numpy as np

    from pct_tools.io import ShuffleMethod, construct_matrix, read_compressed_matrix

    out_dir = tmp_path / "matrices"
    out_dir.mkdir()

    test_matrix_path = out_dir / "test.mat"

    construct_matrix(
        filename=str(test_matrix_path),
        row_indexes=np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
        col_indexes=np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
        values=np.array([1, 1, 1, 1, 2, 2], dtype=np.float32),
        shape=(2, 4),
        compression_level=9,
        shuffle_method=ShuffleMethod.BLOSC_BITSHUFFLE,
        verbose_level=0,
    )
    actual_array = read_compressed_matrix(str(test_matrix_path), 0).toarray()

    expected_array = np.array([[1, 1, 1, 1], [2, 2, 0, 0]], dtype=np.float32)

    np.testing.assert_equal(actual_array, expected_array)


def test_read_compressed_matrix__one_row(tmp_path: Path):
    import numpy as np
    from scipy.sparse import csr_matrix

    from pct_tools.io import ShuffleMethod, construct_matrix, read_compressed_matrix

    out_dir = tmp_path / "matrices"
    out_dir.mkdir()

    test_matrix_path = out_dir / "test.mat"

    construct_matrix(
        filename=str(test_matrix_path),
        row_indexes=np.array([0, 0, 0, 0], dtype=np.int32),
        col_indexes=np.array([0, 1, 2, 3], dtype=np.int32),
        values=np.array([1, 1, 1, 1], dtype=np.float32),
        shape=(1, 4),
        compression_level=9,
        shuffle_method=ShuffleMethod.BLOSC_SHUFFLE,
        verbose_level=0,
    )
    actual_matrix = read_compressed_matrix(str(test_matrix_path), 0)

    expected_matrix = csr_matrix(
        (np.array([1, 1, 1, 1]), (np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3]))),
        shape=(1, 4),
    )

    assert actual_matrix.shape == expected_matrix.shape
    assert actual_matrix.nnz == expected_matrix.nnz

    np.testing.assert_equal(actual_matrix.indices, expected_matrix.indices)
    np.testing.assert_equal(actual_matrix.indptr, expected_matrix.indptr)
    np.testing.assert_equal(actual_matrix.data, expected_matrix.data)


def test_read_compressed_matrix__two_rows(tmp_path: Path):
    import numpy as np
    from scipy.sparse import csr_matrix

    from pct_tools.io import ShuffleMethod, construct_matrix, read_compressed_matrix

    out_dir = tmp_path / "matrices"
    out_dir.mkdir()

    test_matrix_path = out_dir / "test.mat"

    construct_matrix(
        filename=str(test_matrix_path),
        row_indexes=np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
        col_indexes=np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
        values=np.array([1, 1, 1, 1, 2, 2], dtype=np.float32),
        shape=(2, 4),
        compression_level=9,
        shuffle_method=ShuffleMethod.BLOSC_SHUFFLE,
        verbose_level=0,
    )
    actual_matrix = read_compressed_matrix(str(test_matrix_path), 0)

    expected_matrix = csr_matrix(
        (np.array([1, 1, 1, 1, 2, 2]), (np.array([0, 0, 0, 0, 1, 1]), np.array([0, 1, 2, 3, 0, 1]))),
        shape=(2, 4),
    )

    assert actual_matrix.shape == expected_matrix.shape
    assert actual_matrix.nnz == expected_matrix.nnz

    np.testing.assert_equal(actual_matrix.indices, expected_matrix.indices)
    np.testing.assert_equal(actual_matrix.indptr, expected_matrix.indptr)
    np.testing.assert_equal(actual_matrix.data, expected_matrix.data)
