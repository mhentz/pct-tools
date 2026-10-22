"""Tools for proton CT image reconstruction with very large, sparse system matrices.

The package is a thin Python layer over the C++ extension ``pct_tools._ext``. The system
matrix is handled as a set of row blocks, one per block of proton trajectories, each stored
compressed on disk and read back only while it is needed. The submodules are:

* :mod:`pct_tools.matrix`: chord lengths of trajectories through a voxel grid, i.e. the
  entries of the system matrix, and storage of a block as a sparse matrix.
* :mod:`pct_tools.mlp`: most likely path of a proton from its entry and exit measurements.
* :mod:`pct_tools.io`: the binary matrix and vector formats, compressed with Blosc2.
* :mod:`pct_tools.recon`: one steepest-descent step and the squared residual norm over all
  blocks.
* :mod:`pct_tools.lipschitz`: ``A.T @ A @ x`` over all blocks, for the power method.
* :mod:`pct_tools.utils`: recompression and row subsampling of stored blocks.

Submodules are not imported here; import the one you need, e.g.
``from pct_tools.io import read_compressed_matrix``.

Array arguments must be numpy arrays of dtype float32 (int32 for the sparse indices of
:func:`pct_tools.io.construct_matrix`). Other dtypes, and plain lists, raise ``TypeError``
rather than being converted: the C++ kernels work in float32, and a silent conversion would
copy the array, which for a block of trajectories or an image vector can be hundreds of
megabytes. Convert once, deliberately, with ``astype(np.float32)``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pct-tools")
except PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

__all__ = ["__version__", "io", "lipschitz", "matrix", "mlp", "recon", "utils"]
