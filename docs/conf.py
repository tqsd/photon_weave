# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "photon_weave"
copyright = "2024, Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"
author = "Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"
release = "0.0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # for Google-style and NumPy-style docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",  # links to the source code
    "sphinx.ext.autosummary",  # summary tables for documentation
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
autodoc_mock_imports = ["jax", "scipy", "numpy"]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
