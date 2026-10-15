"""Reference rolling results from Python xarray for annoray/crates/annoray-core/tests/rolling.rs.

Run from the repository root:  python3 fixtures/gen_rolling.py > fixtures/rolling.txt

Each line: input|op|window|center|min_periods|shape|values. Inputs are rebuilt in the Rust test.
"""

import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
nan = np.nan

inputs = {
    "f": xr.DataArray([1.0, 2.0, nan, 4.0, 5.0, 6.0, nan, nan, 9.0], dims="x"),
    "i": xr.DataArray(np.array([3, 1, 4, 1, 5, 9, 2], dtype="int32"), dims="x"),
    "g": xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x")),
}
ops = ["sum", "mean", "median", "std", "var", "min", "max", "count", "prod", "argmin", "argmax"]
settings = [(3, False, None), (3, True, None), (4, True, 1), (2, False, 1), (5, False, 2), (1, False, None)]


def fmt(v):
    return " ".join(repr(float(x)) for x in np.asarray(v, dtype=float).ravel())


for name, da in inputs.items():
    for op in ops:
        for w, center, mp in settings:
            r = getattr(da.rolling(x=w, center=center, min_periods=mp), op)()
            print(f"{name}|{op}|{w}|{center}|{mp}|{','.join(map(str, r.shape))}|{fmt(r.values)}")
    for w, center, _ in settings[:3]:
        c = da.rolling(x=w, center=center).construct("w")
        print(f"{name}|construct|{w}|{center}|None|{','.join(map(str, c.shape))}|{fmt(c.values)}")
