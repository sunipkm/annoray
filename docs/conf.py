"""Sphinx configuration for annoray's documentation.

Build locally (after ``maturin develop`` in ``annoray-python``)::

    pip install -r docs/requirements.txt
    sphinx-build -b html -d docs/_build/doctrees docs docs/_build/html

``DOCS_VERSION`` names the version being built (``dev`` or ``X.Y``); the deploy workflow
sets it so the version switcher marks the current page's version.

Versions on the pages are filled in when they are read, so they never go stale: ``@RUST_VERSION@``
(``rust-version`` in the workspace ``Cargo.toml``), ``@PYTHON_VERSION@`` (``requires-python`` in
``annoray-python/pyproject.toml``) and ``@CRATE_VERSION@`` (the requirement that installs the
release being documented: ``X.Y`` for ``X.Y.Z``, the exact version for a pre-release, and for
``dev`` the newest release tag). They are replaced everywhere, code blocks included.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import annoray

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "annoray"
author = "Sunip K. Mukherjee"
copyright = "2026, Sunip K. Mukherjee"
release = annoray.__version__
version = os.environ.get("DOCS_VERSION") or "dev"

# ----- versions on the pages -----

_ROOT = Path(__file__).resolve().parent.parent


def _crate_requirement(v: str) -> str:
    """What a `Cargo.toml` asks for to get release `v`: `X.Y` for `X.Y.Z`, the exact version
    for `X.Y.Z-preN` (Cargo skips pre-releases otherwise)."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(-pre\d+)?", v)
    if m is None:
        raise ValueError(f"not a release version: {v!r}")
    return v if m.group(4) else f"{m.group(1)}.{m.group(2)}"


def _newest_release() -> str:
    """The newest release tag reachable from HEAD, without its `v`."""
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "the dev documentation names the newest release, but no release tag is reachable "
            "(fetch them with `git fetch --tags`, or set DOCS_VERSION=X.Y.Z)"
        ) from e
    return tag.removeprefix("v")


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


_workspace = _toml(_ROOT / "Cargo.toml")["workspace"]["package"]
_pyproject = _toml(_ROOT / "annoray-python" / "pyproject.toml")
PAGE_VERSIONS = {
    "@RUST_VERSION@": _workspace["rust-version"],
    "@PYTHON_VERSION@": _pyproject["project"]["requires-python"].removeprefix(">="),
    "@CRATE_VERSION@": _crate_requirement(version if version != "dev" else _newest_release()),
}


def _fill_versions(app, docname, source):
    for marker, value in PAGE_VERSIONS.items():
        source[0] = source[0].replace(marker, value)


def setup(app):
    app.connect("source-read", _fill_versions)


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

# The executed pages ask for `python3`; they run in a kernel of this interpreter that logs
# only errors. ipykernel warns on every start that its (local, loopback) TCP transport is
# unencrypted, and that would end up in the build output.
_kernels = Path(tempfile.mkdtemp(prefix="annoray-docs-")) / "kernels" / "annoray-docs"
_kernels.mkdir(parents=True)
(_kernels / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
                "--IPKernelApp.log_level=ERROR",
            ],
            "display_name": "Python 3 (annoray docs)",
            "language": "python",
        }
    ),
    encoding="utf-8",
)
os.environ["JUPYTER_PATH"] = os.pathsep.join(
    p for p in (str(_kernels.parent.parent), os.environ.get("JUPYTER_PATH", "")) if p
)
nb_kernel_rgx_aliases = {"python3": "annoray-docs"}

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
