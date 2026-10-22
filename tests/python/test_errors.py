"""Errors raised inside OpenMP parallel regions must surface as Python exceptions."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pair(tmp_path: Path) -> tuple[str, str]:
    from pct_tools.io import construct_matrix, store_vector

    matrix_file = tmp_path / "a.mat"
    vector_file = tmp_path / "b.vec"
    construct_matrix(
        str(matrix_file),
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.ones(4, dtype=np.float32),
        shape=(2, 4),
    )
    store_vector(np.array([1.0, 2.0], dtype=np.float32), str(vector_file))
    return str(matrix_file), str(vector_file)


def test_gradient_descent__wrong_x_size_raises(pair) -> None:
    from pct_tools.recon import gradient_descent

    with pytest.raises(RuntimeError, match="columns but x has"):
        gradient_descent(np.ones(3, dtype=np.float32), 0.1, [pair[0]], [pair[1]], n_threads=4)


def test_squared_residuals__missing_file_raises(pair) -> None:
    from pct_tools.recon import squared_residuals

    with pytest.raises(RuntimeError, match="Could not open file"):
        squared_residuals(
            np.ones(4, dtype=np.float32),
            [pair[0], "/nonexistent.mat"],
            [pair[1], pair[1]],
            n_threads=4,
        )


def test_compute_ATAx__missing_file_raises(pair) -> None:
    from pct_tools.lipschitz import compute_ATAx

    with pytest.raises(RuntimeError, match="Could not open file"):
        compute_ATAx(np.ones(4, dtype=np.float32), [pair[0], "/nonexistent.mat"], n_threads=4)


def test_read_compressed_matrix__truncated_file_raises(pair, tmp_path: Path) -> None:
    from pct_tools.io import read_compressed_matrix

    truncated = tmp_path / "truncated.mat"
    truncated.write_bytes(Path(pair[0]).read_bytes()[:20])
    with pytest.raises(RuntimeError, match="Unexpected end of file"):
        read_compressed_matrix(str(truncated))


def test_construct_matrix__unknown_codec_raises(tmp_path: Path) -> None:
    from pct_tools.io import construct_matrix

    with pytest.raises(ValueError, match="Unknown Blosc codec"):
        construct_matrix(
            str(tmp_path / "x.mat"),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(1, dtype=np.float32),
            shape=(1, 1),
            codec="brotli",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("codec", ["blosclz", "lz4", "lz4hc", "zlib", "zstd"])
def test_construct_matrix__all_codecs_round_trip(tmp_path: Path, codec) -> None:
    from pct_tools.io import construct_matrix, read_compressed_matrix

    filename = str(tmp_path / f"{codec}.mat")
    values = np.arange(1, 7, dtype=np.float32)
    construct_matrix(
        filename,
        np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        np.array([0, 2, 3, 1, 3, 0], dtype=np.int32),
        values,
        shape=(3, 4),
        codec=codec,
    )
    np.testing.assert_array_equal(read_compressed_matrix(filename).data, values)
