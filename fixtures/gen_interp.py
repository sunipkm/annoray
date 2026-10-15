"""Reference interpolation results from Python xarray for annoray/crates/annoray-core/tests/interp.rs.

Run from the repository root:  python3 fixtures/gen_interp.py > fixtures/interp.txt

Each line: case|method|targets|shape|values. Inputs are rebuilt identically in the Rust test.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
nan = np.nan

cases = {
    "nan": xr.DataArray([0.0, 10, nan, 30, 40], dims="x", coords={"x": [0.0, 1, 2, 3, 4]}),
    "uneven": xr.DataArray([1.0, 4, 9, 16, 25], dims="x", coords={"x": [1.0, 2, 3.5, 7, 7.5]}),
    "unsorted": xr.DataArray([1.0, 2, 4], dims="x", coords={"x": [2.0, 0.0, 1.0]}),
    "desc": xr.DataArray([5.0, 3, 1], dims="x", coords={"x": [30.0, 20, 10]}),
    "int": xr.DataArray(np.array([1, 3, 9], dtype="int32"), dims="x", coords={"x": [0.0, 1, 2]}),
    "grid": xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x"), coords={"y": [0.0, 1, 2], "x": [0.0, 10, 20, 30]}),
}
targets = [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3.5, 4, 5, 7.25, 25, nan]


METHODS: tuple[Literal["linear", "nearest"], ...] = ("linear", "nearest")


def fmt(v):
    return " ".join(repr(float(x)) for x in np.asarray(v, dtype=float).ravel())


for name, da in cases.items():
    for method in METHODS:
        r = da.interp(x=targets, method=method)
        print(f"{name}|{method}|x|{','.join(map(str, r.shape))}|{fmt(r.values)}")
g = cases["grid"]
for method in METHODS:
    r = g.interp(x=[5.0, 25.0], y=[0.5, 1.75], method=method)
    print(f"grid2|{method}|x,y|{','.join(map(str, r.shape))}|{fmt(r.values)}")
t = xr.DataArray([0.0, 24.0, 48.0], dims="x", coords={"x": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])})
r = t.interp(x=pd.to_datetime(["2000-01-01T06:00", "2000-01-02T18:00", "2000-01-04T00:00"]))
print(f"time|linear|x|{','.join(map(str, r.shape))}|{fmt(r.values)}")
