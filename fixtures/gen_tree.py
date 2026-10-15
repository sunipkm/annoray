"""Golden DataTree file for annoray/tests/tree.rs, and a checker for ours.

Write the golden file (from the repository root):  python3 fixtures/gen_tree.py
Check a file written by annoray:                  python3 fixtures/gen_tree.py <ours.nc>
(prints OK or raises)

Tree:
  /            coords x (3), attrs title
  /obs         temp(x) using the inherited x; coord station (x) of strings
  /obs/hourly  t (time=2) with its own time coordinate; v(time, x)
  /empty       no variables
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
GOLDEN = os.path.join(os.path.dirname(__file__), "nc", "tree.nc")


def build():
    root = xr.Dataset(coords={"x": [10.0, 20.0, 30.0]}, attrs={"title": "tree"})
    obs = xr.Dataset(
        {"temp": ("x", np.array([1.5, 2.5, 3.5], dtype="float32"), {"units": "K"})},
        coords={"station": ("x", ["a", "bb", "ccc"])},
    )
    hourly = xr.Dataset(
        {"v": (("time", "x"), np.arange(6.0).reshape(2, 3))},
        coords={"time": pd.to_datetime(["2020-01-01T00:00", "2020-01-01T01:00"])},
    )
    return xr.DataTree.from_dict({"/": root, "/obs": obs, "/obs/hourly": hourly, "/empty": xr.Dataset()})


if len(sys.argv) == 1:
    build().to_netcdf(GOLDEN)
    print(xr.open_datatree(GOLDEN))
else:
    golden, ours = xr.open_datatree(GOLDEN), xr.open_datatree(sys.argv[1])
    assert sorted(golden.groups) == sorted(ours.groups), (golden.groups, ours.groups)
    for path in golden.groups:
        a, b = golden[path].to_dataset(), ours[path].to_dataset()
        xr.testing.assert_identical(a, b)
    print("OK")
