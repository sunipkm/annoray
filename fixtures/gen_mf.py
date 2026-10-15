"""Files for open_mfdataset (annoray/tests/mfdataset.rs).

Run from the repository root:  python3 fixtures/gen_mf.py

Writes fixtures/nc/mf/whole.nc and the same dataset split along time into part_*.nc, and
checks that xarray's own open_mfdataset puts the parts back together as the whole.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
OUT = os.path.join(os.path.dirname(__file__), "nc", "mf")
os.makedirs(OUT, exist_ok=True)
time = pd.date_range("2000-01-01", periods=9, freq="D")
ds = xr.Dataset(
    {
        "temp": (("time", "x"), np.arange(27.0).reshape(9, 3), {"units": "K"}),
        "elev": ("x", np.array([10.0, 20.0, 30.0]), {"units": "m"}),
    },
    coords={"time": time, "x": [1.0, 2.0, 3.0]},
    attrs={"title": "mf"},
)
ds.to_netcdf(os.path.join(OUT, "whole.nc"))
for i, sl in enumerate([slice(0, 4), slice(4, 6), slice(6, 9)]):
    ds.isel(time=sl).to_netcdf(os.path.join(OUT, f"part_{i}.nc"))
with xr.set_options(use_new_combine_kwarg_defaults=True):
    back = xr.open_mfdataset(sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("part")))
    xr.testing.assert_identical(back.load(), xr.open_dataset(os.path.join(OUT, "whole.nc")).load())
print("ok")
