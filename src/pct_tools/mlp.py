"""Most likely path (MLP) of a proton through a water-like object.

The MLP is the maximum-likelihood estimate of the transverse position of a proton at each
depth, given its position and direction at the entry and exit detectors. It follows the
formalism of Schulte et al. 2008 (Med. Phys. 35, 4849), in which multiple Coulomb
scattering is modelled by a 2x2 covariance matrix whose elements integrate
``1 / (beta^2 p^2)`` over depth. That integrand is approximated by a 5th-order polynomial
in depth, so each model is described by six coefficients.

Depths and transverse positions are in millimetres, directions in radians. The polynomial
coefficients are given per cm^n, as in the reference, and rescaled to mm internally.

Note:
    The coefficients depend on the beam energy and on the object traversed, since
    ``1 / (beta^2 p^2)`` grows as the proton slows down. The built-in sets returned by
    :func:`get_coefficients` were fitted for one phantom at one energy each and are only
    valid for data acquired under those conditions. For another energy or object, fit your
    own polynomial and pass it to :func:`compute_mlp` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pct_tools._ext

from ._arrays import require_dtype

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_COEFFICIENTS = 6
"""Number of coefficients of the 5th-order polynomial in depth."""


def get_coefficients(phantom: str) -> list[float]:
    """Return the built-in scattering-model coefficients for a named phantom.

    The built-in sets are our own fits to Geant4 simulations of each phantom, following the
    procedure of Schulte et al. 2008; they differ from the values published in that paper.

    Args:
        phantom: ``"ctp528"``, ``"spiral"`` or ``"gammex"``. The CTP528 and spiral phantoms
            share a size and were both simulated at 200 MeV, so they share one set of
            coefficients; the Gammex phantom was simulated at 250 MeV.

    Returns:
        The six coefficients ``c_0 .. c_5`` of the 5th-order polynomial in depth that
        approximates ``1 / (beta^2 p^2)``, with ``c_n`` given per cm^n.

    Raises:
        ValueError: If ``phantom`` is not one of the built-in names.
    """
    return pct_tools._ext.get_coefficients(phantom)


def compute_mlp(
    z: NDArray[np.float32],
    h_in: float,
    theta_in: float,
    h_out: float,
    theta_out: float,
    coefficients: list[float],
    *,
    n_threads: int = 8,
) -> NDArray[np.float32]:
    """Compute the most likely path for a set of scattering-model coefficients.

    Use :func:`get_coefficients` for the built-in models, or supply your own polynomial fit
    of ``1 / (beta^2 p^2)`` against depth, for instance for a different beam energy or
    material.

    Args:
        z: The depths at which to evaluate the path, in mm, of shape ``(n_depths,)``, as a
            float32 array. The first and last entries are the depths of the entry and exit
            detectors.
        h_in: The transverse position at the entry detector, in mm.
        theta_in: The proton direction at the entry detector, in rad.
        h_out: The transverse position at the exit detector, in mm.
        theta_out: The proton direction at the exit detector, in rad.
        coefficients: The six coefficients ``c_0 .. c_5`` of the 5th-order polynomial in
            depth that approximates ``1 / (beta^2 p^2)``, with ``c_n`` given per cm^n.
        n_threads: The number of OpenMP threads used to evaluate the depths in parallel.

    Returns:
        Array of shape ``(n_depths,)`` with the most likely transverse position at each
        depth. The first and last entries equal ``h_in`` and ``h_out``.

    Raises:
        TypeError: If ``z`` is not a float32 array.
        ValueError: If ``coefficients`` does not have exactly six entries.
    """
    require_dtype(z, np.float32, "z")
    if len(coefficients) != N_COEFFICIENTS:
        message = f"The list of coefficients must be of length {N_COEFFICIENTS}"
        raise ValueError(message)

    return pct_tools._ext.compute_mlp(z, h_in, theta_in, h_out, theta_out, coefficients, n_threads)
