"""Sphinx configuration for annoray's documentation.

Build locally (after ``maturin develop`` in ``annoray-python``)::

    pip install -r docs/requirements.txt
    sphinx-build -b html -d docs/_build/doctrees docs docs/_build/html

``DOCS_VERSION`` names the version being built (``dev`` or ``X.Y``); the deploy workflow
sets it so the version switcher marks the current page's version.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import annoray

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "annoray"
author = "Sunip K. Mukherjee"
copyright = "2026, Sunip K. Mukherjee"
release = annoray.__version__
version = os.environ.get("DOCS_VERSION", "dev")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "stubsig",
]

exclude_patterns = ["_build", "README.md", "requirements.txt", "jupyter_execute"]
templates_path = ["_templates"]
html_static_path = ["_static"]

# Docstrings write code in single backticks.
default_role = "code"

# ----- API reference -----

autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "signature"
autodoc_member_order = "groupwise"
autodoc_default_options = {"members": False}
add_module_names = False
# `annoray.dimension` and `annoray.Dimension` would share a file on case-insensitive disks.
autosummary_filename_map = {"annoray.dimension": "annoray.dimension-function"}

# ----- MyST and executed examples -----

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "substitution"]
myst_heading_anchors = 3
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# ----- HTML -----

html_theme = "pydata_sphinx_theme"
html_title = "annoray"
html_theme_options = {
    "github_url": "https://github.com/sunipkm/annoray",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://sunipkm.github.io/annoray/switcher.json",
        "version_match": version,
    },
    "check_switcher": False,
    "show_toc_level": 2,
    # Each API page would otherwise carry the whole reference in its sidebar (~90 KB a page).
    "collapse_navigation": True,
    "navigation_depth": 2,
    "navigation_with_keys": False,
    "external_links": [
        {"name": "Rust API", "url": "https://docs.rs/annoray"},
    ],
}
html_context = {"default_mode": "auto"}
# One page per method: the class pages list them, so their own pages need no section sidebar.
html_sidebars = {"api/generated/*": []}
html_copy_source = False
html_show_sourcelink = False
