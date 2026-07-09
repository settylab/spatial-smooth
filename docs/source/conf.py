"""Sphinx configuration for spatial-smooth."""
from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

from spatial_smooth import __version__  # noqa: E402

project = "spatial-smooth"
author = "Dominik Otto"
copyright = f"{datetime.now():%Y}, Setty Lab, Fred Hutchinson Cancer Center"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Documentation must build without the optional scientific stack installed.
autodoc_mock_imports = ["kompot", "palantir", "squidpy", "KDEpy"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# myst-nb: the tutorial ships pre-executed, so the docs build stays fast and offline.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "colon_fence"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"spatial-smooth {release}"
html_theme_options = {
    "source_repository": "https://github.com/settylab/spatial-smooth/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
suppress_warnings = ["mystnb.unknown_mime_type"]
