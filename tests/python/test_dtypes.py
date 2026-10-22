"""Array arguments must be float32 (int32 for sparse indices); other dtypes raise TypeError.

The bindings would convert any dtype, at the cost of a full copy of the array. The wrappers
reject the conversion so that it happens deliberately on the caller's side. The matrix and
MLP entry points have their own checks in test_matrix.py and test_mlp.py.
"""

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


def test_gradient_descent__float64_x_raises(pair) -> None:
    from pct_tools.recon import gradient_descent

    with pytest.raises(TypeError, match="x must have dtype float32, not float64"):
        gradient_descent(np.ones(4), 0.1, [pair[0]], [pair[1]])


def test_squared_residuals__float64_x_raises(pair) -> None:
    from pct_tools.recon import squared_residuals

    with pytest.raises(TypeError, match="x must have dtype float32, not float64"):
        squared_residuals(np.ones(4), [pair[0]], [pair[1]])


def test_compute_ATAx__float64_x_raises(pair) -> None:
    from pct_tools.lipschitz import compute_ATAx

    with pytest.raises(TypeError, match="x must have dtype float32, not float64"):
        compute_ATAx(np.ones(4), [pair[0]])


def test_construct_matrix__int64_indices_raise(tmp_path: Path) -> None:
    from pct_tools.io import construct_matrix

    with pytest.raises(TypeError, match="row_indexes must have dtype int32, not int64"):
        construct_matrix(
            str(tmp_path / "x.mat"),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.ones(1, dtype=np.float32),
            shape=(1, 1),
        )


def test_construct_matrix__float64_values_raise(tmp_path: Path) -> None:
    from pct_tools.io import construct_matrix

    with pytest.raises(TypeError, match="values must have dtype float32, not float64"):
        construct_matrix(
            str(tmp_path / "x.mat"),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(1),
            shape=(1, 1),
        )


@pytest.mark.parametrize("function", ["store_vector", "store_compressed_vector"])
def test_store_vector__float64_raises(tmp_path: Path, function: str) -> None:
    import pct_tools.io

    with pytest.raises(TypeError, match="vector must have dtype float32, not float64"):
        getattr(pct_tools.io, function)(np.ones(3), str(tmp_path / "v.vec"))


def test_error_message_for_lists_names_the_conversion() -> None:
    from pct_tools.io import store_vector

    with pytest.raises(TypeError, match=r"pass np.asarray\(vector, dtype=np.float32\)"):
        store_vector([1.0, 2.0], "unused.vec")  # type: ignore[arg-type]
