"""Reading and writing sparse matrices and vectors in the package's binary formats.

Matrices are 32-bit float sparse matrices in compressed sparse row (CSR) form with 32-bit
indices; vectors are arrays of 32-bit floats. All integers below are little-endian 32-bit
signed, as written by the C++ code on the platforms the wheels are built for.

Uncompressed matrix (``read_matrix``)::

    rows, cols, nnz, outer_size          4 x int32
    values                               nnz x float32
    outer indices (row pointers)         outer_size x int32
    inner indices (column indices)       nnz x int32

Uncompressed vector (``store_vector`` / ``read_vector``)::

    size                                 int32
    values                               size x float32

Compressed matrix (``store_compressed_matrix`` / ``read_compressed_matrix``)::

    rows, cols, nnz                      3 x int32
    sizes of the three chunks below      3 x int32
    values chunk                         Blosc2 frame of nnz x float32
    outer index chunk                    Blosc2 frame of (rows + 1) x int32
    inner index chunk                    Blosc2 frame of nnz x int32

Compressed vector (``store_compressed_vector`` / ``read_compressed_vector``)::

    size, chunk size                     2 x int32
    values chunk                         Blosc2 frame of size x float32

Each Blosc2 chunk records its own codec and shuffle filter, so files written with any
codec or shuffle method can be read back without specifying them.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

import pct_tools._ext

from ._arrays import require_dtype

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix

    Codec = Literal["blosclz", "lz4", "lz4hc", "zlib", "zstd"]
    """Name of a Blosc2 compression codec."""


class ShuffleMethod(int, Enum):
    """Blosc2 pre-compression filter applied to the raw bytes.

    Shuffling rearranges the bytes (or bits) of each element so that similar bytes become
    contiguous, which usually improves the compression ratio for numerical data.
    """

    BLOSC_NOSHUFFLE = 0
    """Compress the bytes as they are."""
    BLOSC_SHUFFLE = 1
    """Byte shuffling: fast, with a moderate gain in ratio."""
    BLOSC_BITSHUFFLE = 2
    """Bit shuffling: slower, usually the best ratio."""


DEFAULT_CODEC: Codec = "lz4hc"
"""Codec used when none is given: a good ratio with fast decompression."""


def construct_matrix(
    filename: str,
    row_indexes: NDArray[np.int32],
    col_indexes: NDArray[np.int32],
    values: NDArray[np.float32],
    shape: tuple[int, int],
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
    verbose_level: int = 0,
    n_threads: int = 4,
) -> None:
    """Construct a sparse matrix from (row, column, value) triplets and store it compressed.

    Duplicate (row, column) pairs are summed. The shape is given explicitly rather than
    inferred from the indices so that trailing empty rows or columns are preserved.

    Args:
        filename: The filename of the output file.
        row_indexes: The row indices of the matrix elements, an int32 array of shape
            ``(n_elements,)``.
        col_indexes: The column indices of the matrix elements, an int32 array of shape
            ``(n_elements,)``.
        values: The values of the matrix elements, a float32 array of shape ``(n_elements,)``.
        shape: The ``(n_rows, n_cols)`` shape of the resulting matrix.
        compression_level: The compression level 0-9 to use. Default is 9.
        shuffle_method: The blosc shuffle method to use. Default is byte shuffling.
        codec: The Blosc codec to use. Default is LZ4HC.
        verbose_level: The verbosity level.
        n_threads: The number of threads to use.

    Raises:
        TypeError: If the index arrays are not int32 or ``values`` is not float32.
    """
    require_dtype(row_indexes, np.int32, "row_indexes")
    require_dtype(col_indexes, np.int32, "col_indexes")
    require_dtype(values, np.float32, "values")
    pct_tools._ext.construct_matrix(
        filename,
        row_indexes,
        col_indexes,
        values,
        shape[0],
        shape[1],
        compression_level,
        shuffle_method,
        codec,
        verbose_level,
        n_threads,
    )


def read_matrix(filename: str) -> csr_matrix:
    """Read an uncompressed Eigen matrix from disk.

    Args:
        filename: Name of the file containing the matrix.

    Returns:
        The matrix as a sparse matrix of 32-bit floats in CSR format.
    """
    return pct_tools._ext.read_matrix(filename)


def read_compressed_matrix(filename: str, verbosity_level: int = 0) -> csr_matrix:
    """Read compressed Eigen matrix from disk.

    Args:
        filename: Name of the file containing the matrix.
        verbosity_level: The verbosity level.

    Returns:
        The matrix as a sparse matrix of 32-bit floats in CSR format.
    """
    return pct_tools._ext.read_compressed_matrix(filename, verbosity_level)


def store_compressed_matrix(
    matrix: csr_matrix,
    filename: str,
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
    verbosity_level: int = 0,
) -> None:
    """Compress and store a sparse matrix.

    Args:
        matrix: The matrix to store, as a sparse matrix of 32-bit floats in CSR format.
        filename: The filename of the output file.
        compression_level: The compression level 0-9 to use. Default is 9.
        shuffle_method: The blosc shuffle method to use. Default is byte shuffling.
        codec: The Blosc codec to use. Default is LZ4HC.
        verbosity_level: The verbosity level.
    """
    pct_tools._ext.store_compressed_matrix(
        matrix,
        filename,
        compression_level,
        shuffle_method,
        codec,
        verbosity_level,
    )


def read_compressed_vector(filename: str) -> NDArray[np.float32]:
    """Read compressed Eigen vector from disk.

    Args:
        filename: Name of the file containing the vector.

    Returns:
        The vector as a one-dimensional array of 32-bit floats.
    """
    return pct_tools._ext.read_compressed_vector(filename)


def read_vector(filename: str) -> NDArray[np.float32]:
    """Read an Eigen vector from disk.

    Args:
        filename: Name of the file containing the vector.

    Returns:
        The vector as a one-dimensional array of 32-bit floats.
    """
    return pct_tools._ext.read_vector(filename)


def store_compressed_vector(
    vector: NDArray[np.float32],
    filename: str,
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
) -> None:
    """Compress and store a vector.

    Args:
        vector: The one-dimensional float32 array to compress and store.
        filename: The filename of the output file.
        compression_level: The compression level to use.
        shuffle_method: The blosc shuffle method to use. Default is byte shuffling.
        codec: The Blosc codec to use. Default is LZ4HC.

    Raises:
        TypeError: If ``vector`` is not a float32 array.
    """
    require_dtype(vector, np.float32, "vector")
    pct_tools._ext.store_compressed_vector(vector, filename, compression_level, shuffle_method, codec)


def store_vector(vector: NDArray[np.float32], filename: str) -> None:
    """Store a vector.

    Args:
        vector: The one-dimensional float32 array to store.
        filename: The filename of the output file.

    Raises:
        TypeError: If ``vector`` is not a float32 array.
    """
    require_dtype(vector, np.float32, "vector")
    pct_tools._ext.store_vector(vector, filename)
