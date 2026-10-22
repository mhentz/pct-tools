"""Maintenance of stored system-matrix blocks: recompression and row subsampling.

Both operations read a compressed block written by :mod:`pct_tools.io`, transform it on the
C++ side and write a new file, without loading the matrix into Python. Subsampling keeps a
random fraction of the rows of a matrix block and the matching entries of its b-vector, to
emulate a lower dose from the same simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pct_tools._ext
from pct_tools.io import DEFAULT_CODEC, ShuffleMethod

if TYPE_CHECKING:
    from pct_tools.io import Codec


def subsample_matrix_vector_pair(
    input_path: str,
    projection: str,
    output_path: str,
    subsampling_ratio: float,
    pixel_width: float,
    *,
    seed: int = 0,
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
    verbosity_level: int = 0,
) -> None:
    """Subsample matrix and b-vector for a given projection.

    Reads ``<input_path>/matrix-<pixel_width>mm/<projection>.mat`` and
    ``<input_path>/b/b_<projection>.vec``, keeps a random fraction of the rows and writes the
    result to the same relative locations under ``output_path``.

    Args:
        input_path: The path to the input matrices.
        projection: The projection to subsample.
        output_path: The path to the output matrices.
        subsampling_ratio: The fraction of rows to keep, in [0, 1].
        pixel_width: The pixel width in mm; selects the ``matrix-<pixel_width>mm`` directory.
        seed: Seed of the random number generator used to pick rows.
        compression_level: The compression level to use when storing the result.
        shuffle_method: The blosc shuffle method to use.
        codec: The Blosc codec to use. Default is LZ4HC.
        verbosity_level: The verbosity level.
    """
    if seed < 0:
        message = "Seed must be a non-negative integer."
        raise ValueError(message)

    pct_tools._ext.subsample_matrix_vector_pair(
        input_path,
        projection,
        output_path,
        subsampling_ratio,
        f"{pixel_width}mm",
        seed,
        compression_level,
        shuffle_method,
        codec,
        verbosity_level,
    )


def recompress_matrix(
    filename: str,
    output: str,
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
    verbosity_level: int = 0,
) -> None:
    """Recompress a matrix with a given codec and compression level.

    Args:
        filename: The name of the file containing the matrix.
        output: The name of the output file.
        compression_level: The compression level to use when storing it.
        shuffle_method: The blosc shuffle method to use.
        codec: The Blosc codec to use. Default is LZ4HC.
        verbosity_level: The verbosity level.
    """
    pct_tools._ext.recompress_matrix(
        filename,
        output,
        compression_level,
        shuffle_method,
        codec,
        verbosity_level,
    )
