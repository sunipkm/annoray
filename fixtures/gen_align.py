"""Reference alignment results from Python xarray for annoray/crates/annoray-core/tests/align.rs.

Run from the repository root:  python3 fixtures/gen_align.py > fixtures/align.txt

Each line: case|join|object|coord labels|dtype|shape|values (or `error`). Inputs are rebuilt
identically in the Rust test.
"""

import numpy as np
import pandas as pd
import xarray as xr


def da(dims, values, **coords):
    return xr.DataArray(np.array(values), dims=dims, coords=coords)


T = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])

cases = {
    "num": [
        da("x", [10.0, 20.0, 30.0, 40.0], x=[1.0, 2.0, 3.0, 4.0]),
        da("x", np.array([1, 2, 3], dtype="int32"), x=[3.0, 4.0, 5.0]),
    ],
    "unsorted": [
        da("x", [1.0, 2.0, 3.0], x=[3.0, 1.0, 2.0]),
        da("x", [4.0, 5.0, 6.0], x=[2.0, 3.0, 9.0]),
    ],
    "desc": [
        da("x", [1.0, 2.0, 3.0], x=[30.0, 20.0, 10.0]),
        da("x", [4.0, 5.0], x=[20.0, 10.0]),
    ],
    "str": [
        da("s", [1.0, 2.0, 3.0], s=["c", "a", "b"]),
        da("s", [4.0, 5.0], s=["b", "d"]),
    ],
    "time": [
        da("t", [1.0, 2.0, 3.0], t=T[:3]),
        da("t", [4.0, 5.0], t=T[2:]),
    ],
    "equal": [
        da("x", [1.0, 2.0], x=[1.0, 2.0]),
        da("x", [3.0, 4.0], x=[1.0, 2.0]),
    ],
    "three": [
        da("x", [1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0]),
        da("x", [4.0, 5.0, 6.0], x=[2.0, 3.0, 4.0]),
        da("x", [7.0, 8.0], x=[3.0, 2.0]),
    ],
    "nocoord": [
        da(("x", "y"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], x=[1.0, 2.0, 3.0]),
        xr.DataArray(np.array([7.0, 8.0]), dims="y"),
        da("x", [9.0, 10.0], x=[3.0, 1.0]),
    ],
}


def show(name, join, objs):
    try:
        out = xr.align(*objs, join=join)
    except Exception:  # noqa: BLE001 - any failure is an expected error
        print(f"{name}|{join}|error")
        return
    for i, o in enumerate(out):
        dim = o.dims[0]
        labels = " ".join(str(v).replace(" ", "T") for v in np.asarray(o[dim].values).tolist()) if dim in o.coords else "-"
        v = np.asarray(o.values)
        vals = " ".join(repr(float(x)) if v.dtype.kind == "f" else str(x) for x in v.ravel().tolist())
        print(f"{name}|{join}|{i}|{labels}|{v.dtype}|{','.join(map(str, v.shape))}|{vals}")


for name, objs in cases.items():
    for join in ["inner", "outer", "left", "right", "exact"]:
        show(name, join, objs)

# Arithmetic aligns with an inner join.
a, b = cases["num"]
r = a + b
print("arith|+|0|" + " ".join(str(v) for v in r.x.values.tolist()) + f"|{r.dtype}|{r.shape[0]}|" + " ".join(repr(float(x)) for x in r.values))
