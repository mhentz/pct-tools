"""System-matrix construction from proton trajectories.

Each trajectory is a polyline of sampled points ``(x, y, z)``. Its chord length through every
voxel is the length of the polyline inside that voxel, and these chord lengths are the entries
of one row of the system matrix.

Conventions shared by all functions in this module:

* Lengths and coordinates are in the same unit (millimetres in practice); the chord lengths
  come out in that unit too.
* ``img_shape`` is ``(n_i, n_j, n_k)`` and ``coordinate_origin`` is ``(x0, y0, z0)``. A point
  falls into the voxel ``i = floor((y - y0) / slice_thickness)``,
  ``j = floor((x0 - x) / pixel_width)``, ``k = floor((z - z0) / pixel_width)``. Slices are
  therefore stacked along ``y``, ``x`` decreases with increasing ``j`` and ``z`` increases
  with ``k``. The flat voxel index used for matrix columns is ``(i * n_j + j) * n_k + k``,
  i.e. C order of ``img_shape``.
* Each segment between consecutive samples is cut at the voxel boundaries it crosses, and
  every piece contributes its length to the voxel containing it. A segment may cross at most
  one boundary along each axis, so consecutive samples must be less than one voxel apart
  along every axis; a step of half the smallest voxel dimension is a safe choice. Segments
  that cross more raise ``ValueError``. Within that limit the chord lengths are exact for the
  polyline, so the sampling only needs to be fine enough for the polyline to follow the true
  path.
* Pieces that lie outside the image contribute nothing.
* The coordinate arrays must be float32; other dtypes raise ``TypeError`` instead of being
  copied into float32 silently (see the package docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pct_tools._ext

from ._arrays import require_dtype
from .io import DEFAULT_CODEC, ShuffleMethod

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .io import Codec


def compute_matrix_elements(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    img_shape: tuple[int, int, int],
    coordinate_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    *,
    verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.float32]:
    """Compute the dense array of chord lengths for a single proton trajectory.

    Mainly useful for testing and visualisation; use :func:`compute_matrix_triplets` or
    :func:`compute_and_store_sparse_matrix` to build system matrices. See the module
    docstring for the voxel and sampling conventions.

    Args:
        x: The x-coordinates of the sampled trajectory, of shape ``(n_samples,)``.
        y: The y-coordinates of the sampled trajectory, of shape ``(n_samples,)``.
        z: The z-coordinates of the sampled trajectory, of shape ``(n_samples,)``.
        img_shape: The number of voxels ``(n_i, n_j, n_k)`` along y, x and z.
        coordinate_origin: The coordinates ``(x0, y0, z0)`` of the voxel ``(0, 0, 0)`` corner.
        pixel_width: The voxel size along x and z.
        slice_thickness: The voxel size along y.
        verbose: Print debugging information for every segment of the trajectory.
        n_threads: The number of OpenMP threads to use.

    Returns:
        Array of shape ``img_shape`` holding the chord length of the trajectory through each
        voxel, as 32-bit floats.

    Raises:
        TypeError: If ``x``, ``y`` or ``z`` is not a float32 array.
    """
    for name, array in (("x", x), ("y", y), ("z", z)):
        require_dtype(array, np.float32, name)
    return pct_tools._ext.compute_matrix_elements(
        x,
        y,
        z,
        img_shape,
        coordinate_origin,
        pixel_width,
        slice_thickness,
        verbose,
        n_threads,
    )


def compute_matrix_triplets(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    img_shape: tuple[int, int, int],
    coordinate_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    *,
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> list[tuple[int, int, float]]:
    """Compute the non-zero system-matrix entries for a block of proton trajectories.

    See the module docstring for the voxel and sampling conventions.

    Args:
        x: The x-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``
            with one row per trajectory. All trajectories in a block must have the same
            number of samples.
        y: The y-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``.
        z: The z-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``.
        img_shape: The number of voxels ``(n_i, n_j, n_k)`` along y, x and z.
        coordinate_origin: The coordinates ``(x0, y0, z0)`` of the voxel ``(0, 0, 0)`` corner.
        pixel_width: The voxel size along x and z.
        slice_thickness: The voxel size along y.
        verbosity_level: 0 is silent; 1 prints a summary, 2 and 3 print per-block and
            per-segment details.
        n_threads: The number of OpenMP threads to use; trajectories are processed in
            parallel.

    Returns:
        A list of ``(row, column, chord_length)`` triplets, where ``row`` is the trajectory
        index within the block and ``column`` the flat voxel index. A voxel crossed by
        several segments of the same trajectory appears in several triplets; summing
        duplicates is left to the caller (``scipy.sparse`` does so by default).

    Raises:
        TypeError: If ``x``, ``y`` or ``z`` is not a float32 array.
    """
    for name, array in (("x", x), ("y", y), ("z", z)):
        require_dtype(array, np.float32, name)
    return pct_tools._ext.compute_matrix_triplets(
        x,
        y,
        z,
        img_shape,
        coordinate_origin,
        pixel_width,
        slice_thickness,
        verbosity_level,
        n_threads,
    )


def compute_and_store_sparse_matrix(
    filename: str,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    img_shape: tuple[int, int, int],
    coordinate_origin: tuple[float, float, float],
    pixel_width: float,
    slice_thickness: float,
    verbosity_level: int = 0,
    n_threads: int = 8,
    compression_level: int = 9,
    shuffle_method: ShuffleMethod = ShuffleMethod.BLOSC_SHUFFLE,
    codec: Codec = DEFAULT_CODEC,
) -> None:
    """Compute the sparse system matrix for a block of proton trajectories and store it.

    Calls :func:`compute_matrix_triplets`, assembles the triplets into a sparse matrix of
    shape ``(n_trajectories, n_i * n_j * n_k)`` with duplicate entries summed, and stores it
    in the compressed format described in :mod:`pct_tools.io`. Read it back with
    :func:`pct_tools.io.read_compressed_matrix`.

    Args:
        filename: The filename of the output file.
        x: The x-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``
            with one row per trajectory.
        y: The y-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``.
        z: The z-coordinates of the trajectories, of shape ``(n_trajectories, n_samples)``.
        img_shape: The number of voxels ``(n_i, n_j, n_k)`` along y, x and z.
        coordinate_origin: The coordinates ``(x0, y0, z0)`` of the voxel ``(0, 0, 0)`` corner.
        pixel_width: The voxel size along x and z.
        slice_thickness: The voxel size along y.
        verbosity_level: 0 is silent; higher values print progressively more detail.
        n_threads: The number of OpenMP threads to use.
        compression_level: The Blosc compression level 0-9. Default is 9.
        shuffle_method: The Blosc shuffle method. Default is byte shuffling.
        codec: The Blosc codec to use. Default is LZ4HC.

    Raises:
        TypeError: If ``x``, ``y`` or ``z`` is not a float32 array.
    """
    for name, array in (("x", x), ("y", y), ("z", z)):
        require_dtype(array, np.float32, name)
    pct_tools._ext.compute_and_store_sparse_matrix(
        filename,
        x,
        y,
        z,
        img_shape,
        coordinate_origin,
        pixel_width,
        slice_thickness,
        verbosity_level,
        n_threads,
        compression_level,
        shuffle_method,
        codec,
    )
