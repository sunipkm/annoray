"""Toy weather data: climatologies, anomalies, resampling, rolling means, missing values.

After xarray's "Toy weather data" example, with deterministic noise instead of random
numbers so the Rust twin (`annoray/examples/weather.rs`) prints the same lines.

    python examples/weather.py
"""

import numpy as np

import annoray as ar


def fmt(v):
    return f"{float(v):.3f}"


def show(label, values):
    print(f"{label}: [" + ", ".join(fmt(v) for v in np.ravel(values)) + "]")


# Two years of daily minimum and maximum temperatures at three places.
times = ar.date_range("2000-01-01", 731, "D")
doy = ar.DataArray(times, dims="time").dt.dayofyear.values
annual = np.sin(2 * np.pi * (doy / 365.25 - 0.28))
base = 10 + 15 * annual[:, None]
i = np.arange(len(times))[:, None]
j = np.arange(3)[None, :]
tmin = base + 3 * np.sin(0.7 * (7 * i + 13 * j))
tmax = base + 10 + 3 * np.sin(0.7 * (7 * i + 13 * (j + 5)))
tmin[::50, 0] = np.nan  # some missing readings at the first place

ds = ar.Dataset(
    {
        "tmin": (("time", "location"), tmin, {"units": "degC"}),
        "tmax": (("time", "location"), tmax, {"units": "degC"}),
    },
    coords={"time": times, "location": np.array(["IA", "IN", "IL"], dtype=object)},
)
print("sizes: " + ", ".join(f"{k}={n}" for k, n in ds.sizes.items()))

# Missing values: counted, skipped by reductions, or filled.
show("missing tmin per place", ds["tmin"].isnull().sum("time").values)
show("mean tmin (skipping NaN)", ds["tmin"].mean("time").values)

# Monthly climatology, and each day's anomaly from it.
clim = ds["tmin"].groupby("time.month").mean()
show("IA tmin by month", clim.sel(location="IA").values)
anom = ds["tmin"].groupby("time.month") - clim
show("IN anomaly, first week of June 2000",
     anom.sel(location="IN", time=slice("2000-06-01", "2000-06-07")).values)

# Resampling to monthly means, and a centred 7-day rolling mean.
monthly = ds["tmax"].resample(time="MS").mean()
show("IN monthly tmax, 2000 H1", monthly.sel(location="IN").isel(time=slice(0, 6)).values)
weekly = ds["tmax"].rolling(time=7, center=True).mean()
show("IL 7-day tmax, first days", weekly.sel(location="IL").isel(time=slice(0, 6)).values)

# The daily range, and units.
show("mean daily range", (ds["tmax"] - ds["tmin"]).mean("time").values)
show("mean tmax in degF", ds["tmax"].mean("time").to("degF").values)
show("filled tmin, IA, days 0-2", ds["tmin"].fillna(-99.0).sel(location="IA").isel(time=slice(0, 3)).values)
