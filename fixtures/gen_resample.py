"""Reference resample results from Python xarray for xarray-rs/crates/xr-core/tests/resample.rs.

Run from the repository root:  python3 fixtures/gen_resample.py > fixtures/resample.txt

Each line: series|freq|labels|mean|sum|count|first. xarray gives NaN for the sum and count of
an empty bin; xarray-rs gives 0, like any empty sum (and pandas), so those are printed with
NaN replaced by 0. xarray needs a monotonic time index (so no NaT); xarray-rs does not, and
tests that separately.
"""

import warnings

import cftime
import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")

T = [
    "2000-01-01T06:00",
    "2000-01-15T00:00",
    "2000-01-31T00:00",
    "2000-01-31T12:00",
    "2000-02-10T00:00",
    "2000-04-02T00:00",
    "2000-07-01T00:00",
]
series = {
    "std": xr.DataArray(np.arange(7.0), dims="time", coords={"time": pd.to_datetime(T)}),
    "noleap": xr.DataArray(
        np.arange(5.0),
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(2001, 1, 31),
                cftime.DatetimeNoLeap(2001, 2, 28, 12),
                cftime.DatetimeNoLeap(2001, 3, 1),
                cftime.DatetimeNoLeap(2001, 6, 30),
                cftime.DatetimeNoLeap(2002, 1, 1),
            ]
        },
    ),
}
freqs = {
    "std": ["MS", "2MS", "ME", "2ME", "QS", "QE", "QS-FEB", "QE-NOV", "YS", "YE", "W", "2W", "W-MON", "D", "3D", "12h", "5D"],
    "noleap": ["MS", "ME", "QS", "YS", "D"],
}


def fmt(v):
    v = np.asarray(v)
    if v.dtype.kind == "f":
        return " ".join(repr(float(x)) for x in v)
    return " ".join(str(x)[:19].replace(" ", "T") for x in v)


for name, da in series.items():
    for f in freqs[name]:
        r = da.resample(time=f)
        mean = r.mean()
        print(
            "|".join(
                [
                    name,
                    f,
                    fmt(mean.time.values),
                    fmt(mean.values),
                    fmt(r.sum().fillna(0).values),
                    fmt(r.count().fillna(0).values.astype(float)),
                    fmt(r.first().values),
                ]
            )
        )
