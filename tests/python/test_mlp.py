"""Most likely path (MLP) entry points of the Python API.

The model itself is checked against a numerical reference in the C++ suite
(tests/cpp/test_mlp.cpp). These tests cover the wrappers: the named-phantom lookup, argument
validation, result shape and dtype, and two fixed values as a regression guard.
"""

import numpy as np
import pytest

GAMMEX = [4.996e-6, 1.571e-7, -6.656e-9, 1.915e-9, -1.135e-10, 2.724e-12]
CTP528 = [7.512e-6, 3.238e-7, 1.380e-8, 1.339e-9, -1.332e-10, 9.354e-12]

Z = np.array([0.0, 7.0, 31.0, 90.0, 140.0, 199.0, 200.0], dtype=np.float32)
ARGS = (Z, 3.0, -0.02, -1.5, 0.015)


@pytest.mark.parametrize(
    ("phantom", "coefficients"),
    [("gammex", GAMMEX), ("ctp528", CTP528), ("spiral", CTP528)],
)
def test_get_coefficients__built_in_phantoms(phantom, coefficients) -> None:
    from pct_tools.mlp import get_coefficients

    assert get_coefficients(phantom) == coefficients


def test_get_coefficients__unknown_phantom_raises() -> None:
    from pct_tools.mlp import get_coefficients

    with pytest.raises(ValueError, match="water"):
        get_coefficients("water")


def test_compute_mlp__shape_and_dtype() -> None:
    from pct_tools.mlp import compute_mlp

    y = compute_mlp(*ARGS, GAMMEX)

    assert y.shape == Z.shape
    assert y.dtype == np.float32


def test_compute_mlp__straight_line() -> None:
    from pct_tools.mlp import compute_mlp

    actual_y = compute_mlp(
        z=np.arange(5, dtype=np.float32),
        h_in=1,
        theta_in=0,
        h_out=1,
        theta_out=0,
        coefficients=GAMMEX,
    )
    np.testing.assert_allclose(actual_y, np.ones(5, dtype=np.float32), atol=1e-6)


def test_compute_mlp__curve_line() -> None:
    from pct_tools.mlp import compute_mlp

    actual_y = compute_mlp(
        z=np.arange(3, dtype=np.float32),
        h_in=0,
        theta_in=0.785,
        h_out=0,
        theta_out=0,
        coefficients=GAMMEX,
    )
    np.testing.assert_allclose(actual_y, [0, 0.19655585, 0], atol=1e-6)


def test_compute_mlp__phantoms_differ() -> None:
    from pct_tools.mlp import compute_mlp

    assert not np.allclose(compute_mlp(*ARGS, GAMMEX), compute_mlp(*ARGS, CTP528))


def test_compute_mlp__endpoints_are_fixed() -> None:
    from pct_tools.mlp import compute_mlp

    y = compute_mlp(*ARGS, GAMMEX)

    assert y[0] == pytest.approx(ARGS[1])
    assert y[-1] == pytest.approx(ARGS[3])


def test_compute_mlp__wrong_number_of_coefficients_raises() -> None:
    from pct_tools.mlp import compute_mlp

    with pytest.raises(ValueError, match="length 6"):
        compute_mlp(*ARGS, GAMMEX[:5])


def test_compute_mlp__float64_depths_raise() -> None:
    from pct_tools.mlp import compute_mlp

    with pytest.raises(TypeError, match="z must have dtype float32, not float64"):
        compute_mlp(Z.astype(np.float64), *ARGS[1:], GAMMEX)
