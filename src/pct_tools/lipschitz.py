"""Step-size estimation for the reconstruction: the Lipschitz constant of the gradient.

Steepest descent on ``1/2 ||A x - b||^2`` (see :mod:`pct_tools.recon`) is run with step size
``1 / L``, where ``L`` is the Lipschitz constant of the gradient ``A.T @ (A x - b)``. For a
linear least-squares problem ``L`` is the largest eigenvalue of ``A.T @ A``, i.e. the squared
spectral norm of ``A``. It is estimated with the power method, which only needs repeated
products ``A.T @ A @ x``; this module provides that product over a system matrix stored as
row blocks on disk. The power iteration itself is a few lines of numpy on the caller's side,
with ``filenames`` the compressed matrix blocks written by
:func:`pct_tools.matrix.compute_and_store_sparse_matrix` and ``n_voxels`` their number of
columns::

    v = np.random.default_rng(0).random(n_voxels).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(n_iterations):
        v_prev = v
        v = compute_ATAx(v, filenames, n_threads=8)
        v /= np.linalg.norm(v)
        if np.linalg.norm(v - v_prev) < 1e-6:
            break
    L = np.linalg.norm(compute_ATAx(v, filenames, n_threads=8))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pct_tools._ext

from ._arrays import require_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def compute_ATAx(
    x: NDArray[np.float32],
    filenames: Sequence[str],
    *,
    n_threads: int = 1,
) -> NDArray[np.float32]:
    """Compute ``A.T @ A @ x`` for a system matrix stored as row blocks on disk.

    With ``A`` split into blocks ``A_i`` stacked vertically, the result is the sum of
    ``A_i.T @ (A_i @ x)`` over all blocks, which are read and processed in parallel. This is
    the operation needed by the power method to estimate the largest eigenvalue of
    ``A.T @ A``, i.e. the Lipschitz constant of the gradient of ``1/2 ||A x - b||^2``.

    Args:
        x: A one-dimensional float32 array whose length equals the number of columns of
            ``A``.
        filenames: Names of the files holding the compressed matrix blocks.
        n_threads: The number of OpenMP threads to use; each thread handles whole blocks.

    Returns:
        A one-dimensional array of 32-bit floats with the same length as ``x``.

    Raises:
        TypeError: If ``x`` is not a float32 array.
    """
    require_dtype(x, np.float32, "x")
    return pct_tools._ext.compute_ATAx(x, filenames, n_threads)
