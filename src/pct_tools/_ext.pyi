# Type stub for the compiled extension module. The signatures mirror the pybind11
# definitions in src/cpp/bindings.cpp; regenerate the draft with
#     pybind11-stubgen pct_tools._ext
# and keep the array types as ArrayLike / NDArray (pybind11-stubgen emits Eigen shapes that
# are not valid typing, and std::vector parameters accept any array-like). The user-facing
# documentation lives in the Python wrappers.

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

def compute_ATAx(x: ArrayLike, filenames: Sequence[str], n_threads: int = 1) -> NDArray[np.float32]: ...
def compute_and_store_sparse_matrix(
    filename: str,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    img_shape: tuple[int, int, int],
    coord_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    verbosity_level: int = 0,
    n_threads: int = 1,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
) -> None: ...
def compute_matrix_elements(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    img_shape: tuple[int, int, int],
    coord_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    is_verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.float32]: ...
def compute_matrix_triplets(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    img_shape: tuple[int, int, int],
    coord_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> list[tuple[int, int, float]]: ...
def compute_mlp(
    z: ArrayLike,
    h_in: float,
    theta_in: float,
    h_out: float,
    theta_out: float,
    coefficients: Sequence[float],
    n_threads: int = 1,
) -> NDArray[np.float32]: ...
def construct_matrix(
    filename: str,
    rows: ArrayLike,
    cols: ArrayLike,
    values: ArrayLike,
    n_rows: int,
    n_cols: int,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
    verbosity_level: int = 0,
    n_threads: int = 4,
) -> None: ...
def get_coefficients(phantom_type: str) -> list[float]: ...
def gradient_descent(
    x: ArrayLike,
    step_size: float,
    matrix_filenames: Sequence[str],
    b_filenames: Sequence[str],
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> NDArray[np.float32]: ...
def read_compressed_matrix(filename: str, verbosity_level: int = 0) -> csr_matrix: ...
def read_compressed_vector(filename: str) -> NDArray[np.float32]: ...
def read_matrix(filename: str) -> csr_matrix: ...
def read_vector(filename: str) -> NDArray[np.float32]: ...
def recompress_matrix(
    filename: str,
    output: str,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
    verbosity_level: int = 0,
) -> None: ...
def squared_residuals(
    x: ArrayLike,
    matrix_filenames: Sequence[str],
    b_filenames: Sequence[str],
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> float: ...
def store_compressed_matrix(
    matrix: csr_matrix,
    filename: str,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
    verbosity_level: int = 0,
) -> None: ...
def store_compressed_vector(
    vector: ArrayLike,
    filename: str,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
) -> None: ...
def store_vector(vector: ArrayLike, filename: str) -> None: ...
def subsample_matrix_vector_pair(
    input_path: str,
    projection_angle: str,
    output_path: str,
    ratio: float,
    pixel_w_str: str,
    seed: int = 0,
    compression_level: int = 9,
    shuffle_method: int = 1,
    codec: str = "lz4hc",
    verbosity_level: int = 0,
) -> None: ...
