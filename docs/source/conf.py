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
autodoc_mock_imports = ["kompot", "palantir", "scanpy", "squidpy", "KDEpy", "matplotlib"]

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

# GitHub mark, matching the footer icon mellon and kompot use.
_GITHUB_SVG = """
    <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
"""

# No `source_repository`: furo would render "Edit this page" links into a repository that is
# private, and these docs are public -- every such link would 404 for the reader.
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/settylab/spatial-smooth",
            "html": _GITHUB_SVG,
            "class": "",
        },
    ],
}
suppress_warnings = ["mystnb.unknown_mime_type"]
