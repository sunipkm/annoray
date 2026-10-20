"""The examples: annoray works without xarray or pandas, and each Python example prints
exactly what its Rust twin (`annoray/examples/*.rs`) prints."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
REPO = HERE.parents[1]
NAMES = ["quickstart", "weather", "disk_to_disk", "stream_to_file", "quantities"]

# Run code with xarray and pandas unimportable.
BLOCKED = (
    "import sys; sys.modules['xarray'] = None; sys.modules['pandas'] = None; "
    "import runpy; runpy.run_path(sys.argv[1], run_name='__main__')"
)


# The examples print UTF-8 (`°C`, `µm`), as their Rust twins do, on every platform.
UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run_python(name: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", BLOCKED, str(EXAMPLES / f"{name}.py")],
        capture_output=True,
        encoding="utf-8",
        check=False,
        env=UTF8,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_import_does_not_load_xarray():
    code = "import sys, annoray; print('xarray' in sys.modules, 'pandas' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True
    )
    assert out.stdout.strip() == "False False"


@pytest.mark.parametrize("name", NAMES)
def test_examples_run_without_xarray(name):
    assert run_python(name)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
@pytest.mark.parametrize("name", NAMES)
def test_examples_match_rust(name):
    rust = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "annoray", "--example", name],
        cwd=REPO,
        capture_output=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    assert rust.returncode == 0, rust.stderr
    assert run_python(name) == rust.stdout
