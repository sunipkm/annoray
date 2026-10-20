"""Writing over a file that open datasets read from: the file is replaced, never truncated, so
a dataset written back to its own file, and datasets opened before, keep reading the old data
(truncating it under a memory map crashes the process on Linux and macOS, and fails on
Windows)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SCRIPT = """
import sys
import numpy as np
import annoray as ar

path, fmt = sys.argv[1], sys.argv[2]
kw = {"format": "NETCDF3_64BIT"} if fmt == "nc3" else {}
ar.Dataset({"v": ("x", np.arange(100_000.0))}).to_netcdf(path, **kw)

ds = ar.open_dataset(path)
assert ds["v"].is_lazy
ds.to_netcdf(path, **kw)  # back over the file it reads from
assert float(ar.open_dataset(path)["v"].sum()) == 4999950000.0

old = ar.open_dataset(path)
ar.Dataset({"w": ("y", [1.0])}).to_netcdf(path, **kw)
assert float(old["v"].sum()) == 4999950000.0  # the data it opened
assert list(ar.open_dataset(path).data_vars) == ["w"]
print("ok")
"""


@pytest.mark.parametrize("fmt", ["nc4", "nc3"])
def test_overwrite_open_file(tmp_path, fmt):
    # In a child process: a regression would crash it (SIGBUS), not the test session.
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "data.nc"), fmt],
        capture_output=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert out.returncode == 0, (out.returncode, out.stderr)
    assert out.stdout.strip() == "ok"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers
