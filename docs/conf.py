"""Sphinx configuration for the pct-tools API reference.

The pages are generated from the docstrings of the Python wrappers. The compiled extension
``pct_tools._ext`` is mocked, so the documentation builds from the source tree without a C++
toolchain; the wrappers only call into the extension inside function bodies.
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

with (ROOT / "pyproject.toml").open("rb") as f:
    _project = tomllib.load(f)["project"]

project = "pct-tools"
author = _project["authors"][0]["name"]
release = version = _project["version"]
project_copyright = f"Matthieu Hentz. Version {release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autodoc_mock_imports = ["pct_tools._ext"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {"members": True, "undoc-members": False}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 2

templates_path: list[str] = []
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"pct-tools {release}"
html_theme_options = {
    "source_repository": "https://github.com/mhentz/pct-tools",
    "source_branch": "main",
    "source_directory": "docs/",
}
