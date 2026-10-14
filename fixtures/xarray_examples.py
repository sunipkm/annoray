"""Reference results from Python xarray for xarray-rs/tests/xarray_examples.rs.

Run from the repository root:  python3 fixtures/xarray_examples.py
Each line prints a case name and the resulting values; the Rust tests hardcode them.
"""

import numpy as np
import pandas as pd
import xarray as xr

time = pd.to_datetime(["2020-01-01", "2020-01-15", "2020-02-01", "2020-03-01"])
temp = xr.DataArray(
    np.arange(12.0).reshape(4, 3),
    dims=("time", "lat"),
    coords={"time": time, "lat": [10.0, 20.0, 30.0]},
    name="temp",
)
desc = temp.isel(lat=slice(None, None, -1))


def show(name, da):
    v = da.values
    print(f"{name}: dims={da.dims} values={v.ravel().tolist()}")


def show_err(name, f):
    try:
        f()
        print(f"{name}: no error")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")


print(f"# xarray {xr.__version__}")
show("sel_lat_20", temp.sel(lat=20.0))
show("sel_lat_15_30", temp.sel(lat=slice(15, 30)))
show("sel_desc_30_15", desc.sel(lat=slice(30, 15)))
show("sel_time_month", temp.sel(time="2020-01"))
show("sel_time_range", temp.sel(time=slice("2020-01-10", "2020-02")))
show("nearest_24", temp.sel(lat=24, method="nearest"))
show_err("nearest_24_tol3", lambda: temp.sel(lat=24, method="nearest", tolerance=3))
show("pad_25", temp.sel(lat=25, method="pad"))
show("backfill_25", temp.sel(lat=25, method="backfill"))
show("desc_pad_25", desc.sel(lat=25, method="pad"))
show("desc_backfill_25", desc.sel(lat=25, method="backfill"))
show("nearest_time", temp.sel(time="2020-01-20", method="nearest"))
show("isel_last_rev", temp.isel(time=-1, lat=slice(None, None, -1)))
show("sel_step", temp.sel(time=slice(None, None, 2), lat=slice(10, 30, 2)))
show_err("missing_label", lambda: temp.sel(lat=25.0))

lat_w = xr.DataArray([1.0, 2.0, 3.0], dims="lat", coords={"lat": [10.0, 20.0, 30.0]})
show("broadcast_add", temp + lat_w)
other = xr.DataArray([1.0, 2.0], dims="x")
show("outer_broadcast", lat_w * other)
d = temp.time - temp.time.isel(time=0)
print(f"time_diff_days: {(d.values / np.timedelta64(1, 'D')).tolist()}")
ints = xr.DataArray(np.array([7, -7, 9], dtype="int32"), dims="x")
show("int_div", ints / xr.DataArray(np.array([2, 2, 3], dtype="int32"), dims="x"))
show("int_mul", ints * xr.DataArray(np.array([2, 3, 4], dtype="int32"), dims="x"))
show("isel_list", temp.isel(time=[3, 0, -1]))
show("sel_list", temp.sel(lat=[30.0, 10.0]))
show("sel_list_nearest", temp.sel(lat=[29.0, 11.0], method="nearest"))
show_err("isel_list_oob", lambda: temp.isel(time=[4]))
