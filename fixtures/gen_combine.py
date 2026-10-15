"""Reference concat/merge results from Python xarray for annoray/crates/annoray-core/tests/combine.rs.

Run from the repository root:  python3 fixtures/gen_combine.py > fixtures/combine.txt

Each case prints `case|mode` and then one line per variable (coordinates first, sorted by
name within each group): `  name|coord or var|dims|dtype|values`; or `case|mode|error`.
`mode` is `new` (xarray's new combine defaults) or `legacy` (the current ones).
"""

import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
nan = np.nan


def ds(time=None, x=(10.0, 20.0, 30.0), temp=None, const=(1.0, 2.0, 3.0), **extra):
    coords = {"x": list(x)}
    data = {}
    if time is not None:
        coords["time"] = list(time)
        data["temp"] = (("time", "x"), np.array(temp))
    else:
        data["temp"] = (("x",), np.array(temp))
    if const is not None:
        data["const"] = (("x",), np.array(const))
    for k, v in extra.items():
        data[k] = v
    return xr.Dataset(data, coords=coords)


A = ds(time=[1.0, 2.0], temp=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = ds(time=[3.0], temp=[[7.0, 8.0, 9.0]])
C = ds(time=[3.0], temp=[[7.0, 8.0, 9.0]], const=(9.0, 9.0, 9.0))
D = ds(time=[3.0], x=(20.0, 30.0, 40.0), temp=[[7.0, 8.0, 9.0]])
E = ds(time=[3.0], temp=np.array([[7, 8, 9]], dtype="int32"))
P = ds(temp=[1.0, 2.0, 3.0])
Q = ds(temp=[4.0, 5.0, 6.0])

concat_cases = {
    "time": ([A, B], "time"),
    "const_differs": ([A, C], "time"),
    "misaligned": ([A, D], "time"),
    "dtype": ([A, E], "time"),
    "new_dim": ([P, Q], "run"),
}

M1 = xr.Dataset({"a": ("x", [1.0, nan, 3.0])}, coords={"x": [1.0, 2.0, 3.0]})
M2 = xr.Dataset({"a": ("x", [nan, 2.0, 3.0]), "b": ("x", [4.0, 5.0, 6.0])}, coords={"x": [1.0, 2.0, 3.0]})
M3 = xr.Dataset({"a": ("x", [9.0, 2.0, 3.0])}, coords={"x": [1.0, 2.0, 3.0]})
M4 = xr.Dataset({"c": ("x", [7.0, 8.0])}, coords={"x": [2.0, 4.0]})

merge_cases = {
    "fill": [M1, M2],
    "conflict": [M1, M3],
    "misaligned": [M1, M4],
}


def show(name, mode, f):
    with xr.set_options(use_new_combine_kwarg_defaults=(mode == "new")):
        try:
            out = f()
        except Exception:  # noqa: BLE001 - any failure is an expected error
            print(f"{name}|{mode}|error")
            return
    print(f"{name}|{mode}")
    for kind, names in (("coord", sorted(out.coords)), ("var", sorted(out.data_vars))):
        for n in names:
            v = out[n].variable
            vals = " ".join(repr(float(x)) if v.dtype.kind == "f" else str(x) for x in np.asarray(v.values).ravel().tolist())
            print(f"  {n}|{kind}|{','.join(v.dims)}|{v.dtype}|{vals}")


for mode in ("new", "legacy"):
    for name, (objs, dim) in concat_cases.items():
        show(f"concat_{name}", mode, lambda objs=objs, dim=dim: xr.concat(objs, dim))
    for name, objs in merge_cases.items():
        show(f"merge_{name}", mode, lambda objs=objs: xr.merge(objs))
