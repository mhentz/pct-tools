"""Steepest descent on ``1/2 ||A x - b||^2`` for a system stored as blocks on disk.

The system matrix ``A`` and the measurement vector ``b`` are split into row blocks
``(A_i, b_i)``, one pair of files per projection, written by :mod:`pct_tools.matrix` and
the preprocessing. Each call here streams through every block in parallel and returns one
quantity for the whole system: one descent step, or the squared residual norm. The iteration
itself, including the step size (see :mod:`pct_tools.lipschitz`) and the stopping rule, is
left to the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pct_tools._ext

from ._arrays import require_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def gradient_descent(
    x: NDArray[np.float32],
    step_size: float,
    matrix_filenames: Sequence[str],
    vector_filenames: Sequence[str],
    *,
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> NDArray[np.float32]:
    """Perform one steepest-descent step on ``1/2 ||A x - b||^2``.

    The system is split into blocks ``(A_i, b_i)`` that are stored on disk and processed in
    parallel; the result is ``x - step_size * sum_i A_i^T (A_i x - b_i)``.

    Args:
        x: The current estimate, a float32 array of shape ``(n_voxels,)`` where ``n_voxels``
            is the number of columns of ``A``.
        step_size: The step size to be used in the descent, typically ``1 / L`` with ``L`` the
            Lipschitz constant estimated via :mod:`pct_tools.lipschitz`.
        matrix_filenames: The filenames of the compressed matrix blocks.
        vector_filenames: The filenames of the matching b-vector blocks.
        verbosity_level: The verbosity level.
        n_threads: The number of threads to use.

    Returns:
        The updated vector after one step of steepest descent, with the same shape as ``x``.

    Raises:
        TypeError: If ``x`` is not a float32 array.
    """
    require_dtype(x, np.float32, "x")
    return pct_tools._ext.gradient_descent(
        x,
        step_size,
        matrix_filenames,
        vector_filenames,
        verbosity_level,
        n_threads,
    )


def squared_residuals(
    x: NDArray[np.float32],
    matrix_filenames: Sequence[str],
    vector_filenames: Sequence[str],
    *,
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> float:
    """Compute the squared residual norm ``||A x - b||^2`` for a given vector.

    Args:
        x: The current estimate, a float32 array of shape ``(n_voxels,)``.
        matrix_filenames: The filenames of the compressed matrix blocks.
        vector_filenames: The filenames of the matching b-vector blocks.
        verbosity_level: The verbosity level.
        n_threads: The number of threads to use.

    Returns:
        The squared residual norm.

    Raises:
        TypeError: If ``x`` is not a float32 array.
    """
    require_dtype(x, np.float32, "x")
    return pct_tools._ext.squared_residuals(x, matrix_filenames, vector_filenames, verbosity_level, n_threads)
