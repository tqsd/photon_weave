# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "photon_weave"
copyright = "2024, Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"
author = "Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel"
release = "0.0.3"
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx_autodoc_typehints",  # Show type hints in function signatures
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
