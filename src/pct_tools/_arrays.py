"""Input-array checks shared by the Python wrappers.

The C++ kernels work on float32 data (int32 for sparse indices). pybind11 would convert any
other dtype on entry, but that conversion allocates a full copy of the array, which for a
block of trajectories or an image vector can be hundreds of megabytes. The wrappers
therefore reject other dtypes instead of converting silently, so that the copy, if one is
needed, is made deliberately by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def require_dtype(array: object, dtype: DTypeLike, name: str) -> None:
    """Raise ``TypeError`` unless ``array`` is a numpy array whose dtype is exactly ``dtype``.

    Args:
        array: The argument to check.
        dtype: The required dtype, e.g. ``np.float32``.
        name: The parameter name, used in the error message.

    Raises:
        TypeError: If ``array`` is not a :class:`numpy.ndarray` or has another dtype.
    """
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        message = (
            f"{name} must be a numpy array of dtype {expected}, not {type(array).__name__}; "
            f"pass np.asarray({name}, dtype=np.{expected})"
        )
        raise TypeError(message)
    if array.dtype != expected:
        message = (
            f"{name} must have dtype {expected}, not {array.dtype}; pass {name}.astype(np.{expected}). "
            "The conversion is left to the caller because it copies the array."
        )
        raise TypeError(message)
