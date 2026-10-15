"""Disk to disk: compressed netCDF out, lazy reads, streamed computation and reductions.

Data opened from files stays on disk: selections, arithmetic, unit conversion and
reductions read it chunk by chunk, so it never has to fit in memory. The Rust twin is
`annoray/examples/disk_to_disk.rs`; both print the same lines.

    python examples/disk_to_disk.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

import annoray as ar


def fmt(v):
    return f"{float(v):.3f}"


def yes(b):
    return "yes" if b else "no"


def show(label, values):
    print(f"{label}: [" + ", ".join(fmt(v) for v in np.ravel(values)) + "]")


tmp = Path(tempfile.mkdtemp(prefix="annoray-example-"))
try:
    # Two years of daily air temperature on a small grid, in kelvin.
    times = ar.date_range("2000-01-01", 731, "D")
    t = np.arange(731)[:, None, None]
    lat = np.linspace(-45.0, 45.0, 20)
    lon = np.arange(0.0, 360.0, 12.0)
    air = (
        288.0
        - 30.0 * np.abs(np.sin(np.radians(lat)))[None, :, None]
        + 8.0 * np.sin(2 * np.pi * t / 365.25)
        + np.cos(np.radians(lon))[None, None, :]
    )
    ds = ar.Dataset(
        {"air": (("time", "lat", "lon"), air, {"units": "K"})},
        coords={"time": times, "lat": lat, "lon": lon},
    )

    # Compressed, chunked netCDF-4; and one file per year.
    enc = {"air": {"zlib": True, "complevel": 4, "chunksizes": (30, 20, 30)}}
    ds.to_netcdf(tmp / "air.nc", encoding=enc)
    ds.isel(time=slice(0, 366)).to_netcdf(tmp / "air_2000.nc", encoding=enc)
    ds.isel(time=slice(366, 731)).to_netcdf(tmp / "air_2001.nc", encoding=enc)
    print("wrote air.nc, air_2000.nc, air_2001.nc")

    # Opening reads metadata only; values stay on disk until used.
    on_disk = ar.open_dataset(tmp / "air.nc")
    print(f"opened: lazy={yes(on_disk['air'].is_lazy)}")

    # Arithmetic and unit conversion stay lazy; writing computes chunk by chunk.
    celsius = on_disk["air"].to("degC")
    anomaly = celsius - celsius.mean("time")
    print(f"anomaly: lazy={yes(anomaly.is_lazy)}, units={anomaly.units}")
    ar.Dataset({"anomaly": anomaly}).to_netcdf(tmp / "anomaly.nc")
    back = ar.open_dataset(tmp / "anomaly.nc")["anomaly"]
    show("anomaly at (lat 0, lon 0), first days", back.isel(lat=10, lon=0, time=slice(0, 4)).values)

    # Reductions of on-disk data stream through a memory budget (here 1 MiB at a time).
    ar.set_stream_bytes(1 << 20)
    show("time-mean air, lat 0..3, lon 0", on_disk["air"].mean("time").isel(lon=0, lat=slice(0, 3)).values)
    show("time-std air, lat 0..3, lon 0", on_disk["air"].std("time").isel(lon=0, lat=slice(0, 3)).values)
    monthly = on_disk["air"].groupby("time.month").mean()
    show("monthly mean at (lat 10, lon 0)", monthly.isel(lat=10, lon=0).values)

    # Many files as one dataset, combined by their coordinates.
    both = ar.open_mfdataset(str(tmp / "air_*.nc"))
    print("multi-file sizes: " + ", ".join(f"{k}={n}" for k, n in both.sizes.items()))
    show("overall mean (one file, two files)", [on_disk["air"].mean().values, both["air"].mean().values])
finally:
    shutil.rmtree(tmp)
