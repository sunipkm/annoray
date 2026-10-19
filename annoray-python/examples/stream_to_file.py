"""Streaming to a new file: a netCDF-4 file made from a template and filled in as the values
are produced, readable while it is written.

``NetcdfWriter.create`` writes the template's dimensions, variables, attributes, encodings and
coordinates; the data variables read as fill values until written. Each step is appended along
the unlimited ``time`` dimension, a region is rewritten in place, and the file is read back at
any point. The Rust twin is `annoray/examples/stream_to_file.rs`; both print the same lines.

    python examples/stream_to_file.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

import annoray as ar

NLAT, NLON = 4, 6


def show(label, values):
    print(f"{label}: [" + ", ".join(f"{float(v):.2f}" for v in np.ravel(values)) + "]")


def fields(first, hours):
    """`hours` hourly fields of temperature from hour `first`, in kelvin."""
    h = np.arange(first, first + hours)[:, None, None]
    la = np.arange(NLAT)[None, :, None]
    lo = np.arange(NLON)[None, None, :]
    temp = (280.0 + h + 0.5 * la + 0.1 * lo).astype("f4")
    time = np.datetime64("2000-01-01T00", "ns") + np.arange(first, first + hours) * np.timedelta64(
        1, "h"
    )
    return ar.Dataset(
        {"temp": (("time", "lat", "lon"), temp, {"units": "K"})},
        coords={"time": ("time", time)},
    )


tmp = Path(tempfile.mkdtemp(prefix="annoray-stream-"))
try:
    path = tmp / "run.nc"

    # The template: no time steps yet, the grid's coordinates, and how to store the values.
    template = fields(0, 0).assign_coords(
        lat=("lat", -30.0 + 20.0 * np.arange(NLAT)), lon=("lon", 60.0 * np.arange(NLON))
    )
    encoding = {"time": {"units": "hours since 2000-01-01"}, "temp": {"zlib": True, "complevel": 4}}

    with ar.NetcdfWriter.create(path, template, unlimited_dims=["time"], encoding=encoding) as w:
        print(f"created run.nc: time={w.dim_len('time')} (unlimited), lat={NLAT}, lon={NLON}")

        # Each hour's field is appended as soon as it exists.
        for hour in range(6):
            w.append(fields(hour, 1), "time")
            if hour == 2:
                # The file so far, opened lazily while writing goes on.
                now = w.dataset()
                show("after 3 hours, temp at (lat 0, lon 0)", now["temp"].isel(lat=0, lon=0).values)
        print(f"appended 6 hours: time={w.dim_len('time')}")

        # A correction to part of hour 2: the region's own coordinates are left as they are.
        warm = np.full((1, NLAT, 3), 300.0, "f4")
        fix = ar.Dataset({"temp": (("time", "lat", "lon"), warm, {"units": "K"})})
        w.write(fix, region={"time": 2, "lon": slice(3, 6)})
        print("rewrote hour 2 at lon 3..6")

    back = ar.open_dataset(path)
    show("temp at (lat 1, lon 4), hour by hour", back["temp"].isel(lat=1, lon=4).values)
    print(
        f"on disk: time={back.sizes['time']}, units={back['temp'].units}, "
        f"time units={back['time'].encoding['units']}"
    )
finally:
    shutil.rmtree(tmp, ignore_errors=True)
