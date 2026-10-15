"""Reference reductions from Python xarray for annoray/crates/annoray-core/tests/reductions.rs.

Run from the repository root:  python3 fixtures/gen_reductions.py > fixtures/reductions.txt

Each line: case|op|dims|skipna|ddof|min_count|dtype|shape|values (or `error`). Inputs are
rebuilt identically in the Rust test.
"""

import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
nan = np.nan

F = [[1.0, nan, 3.0, 4.0], [nan, nan, nan, nan], [2.5, -1.0, 0.0, 7.0]]
I = [[1, -2, 3, 4], [5, 6, -7, 8], [9, 10, 11, -12]]
U = [[1, 2, 3, 4], [250, 6, 7, 8], [0, 0, 0, 1]]
B = [[True, False, True, True], [False, False, False, False], [True, True, True, True]]
T = np.array(
    ["2020-01-01T00:00", "2021-06-15T12:00", "1999-12-31T23:59:59.5", "2020-01-02"],
    dtype="datetime64[ns]",
)
D = np.array([1_000_000_000, -5, 3_600_000_000_000, 7], dtype="timedelta64[ns]")

inputs = {
    "f64": xr.DataArray(np.array(F, dtype="float64"), dims=("x", "y")),
    "f32": xr.DataArray(np.array(F, dtype="float32"), dims=("x", "y")),
    "i32": xr.DataArray(np.array(I, dtype="int32"), dims=("x", "y")),
    "u8": xr.DataArray(np.array(U, dtype="uint8"), dims=("x", "y")),
    "bool": xr.DataArray(np.array(B), dims=("x", "y")),
    "time": xr.DataArray(T, dims=("t",)),
    "delta": xr.DataArray(D, dims=("t",)),
}

OPS = ["sum", "prod", "mean", "median", "std", "var", "min", "max", "count", "argmin", "argmax", "any", "all"]


def fmt(v, dtype):
    if dtype.kind == "f":
        return repr(float(v))
    if dtype.kind == "M":
        return str(np.datetime64(v, "ns"))
    if dtype.kind == "m":
        return str(int(np.timedelta64(v, "ns").astype("int64")))
    if dtype.kind == "b":
        return "true" if v else "false"
    return str(int(v))


def run(name, da, op, dims, skipna=None, ddof=None, min_count=None):
    kw = {}
    if skipna is not None:
        kw["skipna"] = skipna
    if ddof is not None:
        kw["ddof"] = ddof
    if min_count is not None:
        kw["min_count"] = min_count
    if op in ("any", "all", "count"):
        kw.pop("skipna", None)
    dim_arg = None if dims == "all" else dims.split(",")
    if op in ("argmin", "argmax") and dim_arg is not None:
        dim_arg = dim_arg[0]
    try:
        r = getattr(da, op)(dim=dim_arg, **kw)
        v = np.asarray(r.values)
        out = f"{v.dtype}|{','.join(map(str, v.shape))}|" + " ".join(fmt(x, v.dtype) for x in v.ravel())
    except Exception:  # noqa: BLE001 - any failure is an expected error
        out = "error"
    s = "" if skipna is None else str(skipna).lower()
    d = "" if ddof is None else str(ddof)
    m = "" if min_count is None else str(min_count)
    print(f"{name}|{op}|{dims}|{s}|{d}|{m}|{out}")


for name, da in inputs.items():
    dim_sets = ["all", "t"] if da.ndim == 1 else ["all", "x", "y", "y,x"]
    for op in OPS:
        for dims in dim_sets:
            if op in ("argmin", "argmax") and dims in ("all", "y,x") and da.ndim > 1:
                continue
            run(name, da, op, dims)
        if name in ("f64", "f32"):
            for dims in dim_sets:
                if op in ("argmin", "argmax") and dims in ("all", "y,x"):
                    continue
                if op not in ("any", "all", "count"):
                    run(name, da, op, dims, skipna=False)
            if op in ("std", "var"):
                for dims in dim_sets:
                    run(name, da, op, dims, ddof=1)
            if op in ("sum", "prod"):
                for dims in dim_sets:
                    run(name, da, op, dims, min_count=2)
        if name in ("i32",) and op in ("std", "var"):
            run(name, da, op, "y", ddof=1)
