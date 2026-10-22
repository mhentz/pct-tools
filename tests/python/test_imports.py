"""Smoke tests: the extension module and every wrapper module can be imported.

A failure here usually means the extension did not build, was built against another Python,
or a shared library (OpenMP, Blosc2) is missing at runtime.
"""


def test_imports__ext() -> None:
    import pct_tools._ext

    assert pct_tools._ext


def test_imports__io() -> None:
    import pct_tools.io

    assert pct_tools.io


def test_imports__lipschitz() -> None:
    import pct_tools.lipschitz

    assert pct_tools.lipschitz


def test_imports__matrix() -> None:
    import pct_tools.matrix

    assert pct_tools.matrix


def test_imports__mlp() -> None:
    import pct_tools.mlp

    assert pct_tools.mlp


def test_imports__recon() -> None:
    import pct_tools.recon

    assert pct_tools.recon


def test_imports__utils() -> None:
    import pct_tools.utils

    assert pct_tools.utils
