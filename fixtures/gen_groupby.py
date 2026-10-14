"""Reference groupby results from Python xarray for xarray-rs/crates/xr-core/tests/groupby.rs.

Run from the repository root:  python3 fixtures/gen_groupby.py > fixtures/groupby.txt

Each line: case|dims|labels|values. Inputs are rebuilt identically in the Rust test.
Missing labels are covered by `num` (NaN); Rust strings have no missing value. For
`include_lowest`, pandas labels the first bin "(9.999, 30.0]"; xarray-rs says "[10, 30]".
"""

import warnings

import cftime
import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
nan = np.nan

lab = xr.DataArray(
    [1.0, 2, 3, 4, 5, 6],
    dims="x",
    coords={"x": [10, 20, 30, 40, 50, 60], "lab": ("x", ["b", "a", "b", "c", "c", "a"])},
)
num = xr.DataArray([1.0, 2, 3, 4], dims="x", coords={"k": ("x", [2.0, nan, 1.0, 2.0])})
time = pd.date_range("2000-01-01", periods=24, freq="MS")
grid = xr.DataArray(
    np.arange(48.0).reshape(24, 2), dims=("time", "y"), coords={"time": time, "y": [0, 1]}
)
noleap = xr.DataArray(
    np.arange(6.0),
    dims="time",
    coords={"time": [cftime.DatetimeNoLeap(2001, m, 15) for m in (1, 2, 3, 1, 2, 3)]},
)


def fmt(v):
    v = np.asarray(v)
    if v.dtype.kind == "f":
        return " ".join(repr(float(x)) for x in v.ravel())
    return " ".join(str(x) for x in v.ravel())


def show(name, r, label):
    labels = fmt(r[label].values) if label else "-"
    print(f"{name}|{','.join(r.dims)}|{labels}|{fmt(r.values)}")


show("lab_mean", lab.groupby("lab").mean(), "lab")
show("lab_sum", lab.groupby("lab").sum(), "lab")
show("lab_max", lab.groupby("lab").max(), "lab")
show("lab_count", lab.groupby("lab").count(), "lab")
show("lab_first", lab.groupby("lab").first(), "lab")
show("lab_last", lab.groupby("lab").last(), "lab")
show("num_sum", num.groupby("k").sum(), "k")
show("bins_mean", lab.groupby_bins("x", [0, 25, 45, 100, 200]).mean(), "x_bins")
show("bins_left", lab.groupby_bins("x", [10, 30, 60], right=False).sum(), "x_bins")
show("bins_lowest", lab.groupby_bins("x", [10, 30, 60], include_lowest=True).sum(), "x_bins")
show("month_mean", grid.groupby("time.month").mean(), "month")
show("season_mean", grid.groupby("time.season").mean(), "season")
show("year_sum", grid.groupby("time.year").sum(), "year")
show("noleap_month", noleap.groupby("time.month").mean(), "month")
clim = grid.groupby("time.season").mean()
show("anomaly", grid.groupby("time.season") - clim, "season")
show("map_anomaly", grid.groupby("time.month").map(lambda g: g - g.mean("time")), None)
show("map_reduce", grid.groupby("time.year").map(lambda g: g.max("time")), "year")
