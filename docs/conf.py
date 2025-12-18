# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback for local builds
    import tomli as tomllib

sys.path.insert(0, os.path.abspath(".."))

project = "photon_weave"
copyright = "2025, Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"
author = "Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"

# Keep the docs version in sync with the package metadata
_pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
with _pyproject_path.open("rb") as f:
    _pyproject = tomllib.load(f)

release = _pyproject["tool"]["poetry"]["version"]
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # for Google-style and NumPy-style docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",  # links to the source code
    "sphinx.ext.autosummary",  # summary tables for documentation
    "myst_parser",  # markdown support
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
autodoc_mock_imports = [
    "jax",
    "scipy",
    "numpy",
    # keep heavyweight external deps mocked; project modules are real
]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
