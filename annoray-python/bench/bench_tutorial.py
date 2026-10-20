"""Benchmarks on xarray's tutorial datasets: I/O per engine, and common workflows with
xarray vs native annoray. Every annoray result is checked against xarray's before timing.

    python bench_tutorial.py [--repeat N] [--json out.json]
"""

from __future__ import annotations

import os

# scipy (imported by xarray.tutorial) starts OpenBLAS threads that spin idle and take cores
# from both libraries; one is enough here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import statistics
import sys
import time
import warnings

import numpy as np
import xarray as xr

import annoray as ar

warnings.filterwarnings("ignore")


def timeit(f, repeat):
    f()  # warm-up (and caches)
    ts = []
    for _ in range(repeat):
        s = time.perf_counter()
        f()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


def same(r, x, what):
    """annoray's result must match xarray's (values and coordinates)."""
    got = r.to_xarray() if hasattr(r, "to_xarray") else r
    if isinstance(got, xr.DataArray) and isinstance(x, xr.DataArray):
        got = got.rename(x.name)
    # float32 data: xarray accumulates in float32 (errors ~1e-5 of the values); annoray in
    # float64. Allow for xarray's rounding there.
    f32 = getattr(x, "dtype", None) == np.float32 or (
        isinstance(x, xr.Dataset) and any(v.dtype == np.float32 for v in x.data_vars.values())
    )
    try:
        xr.testing.assert_allclose(got, x, rtol=1e-5, atol=1e-4 if f32 else 1e-8)
    except AssertionError as e:
        raise AssertionError(f"{what}: results differ\n{str(e)[:600]}") from None


def path(name):
    return str(xr.tutorial.open_dataset(name).encoding["source"])


# ----- I/O -----


def io_rows(names, repeat):
    rows = []
    for name in names:
        p = path(name)
        size = os.path.getsize(p) / 1e6
        ref = xr.open_dataset(p, engine="netcdf4").load()
        times = {}
        for eng in ["netcdf4", "h5netcdf", "annoray"]:
            try:
                xr.testing.assert_identical(xr.open_dataset(p, engine=eng).load(), ref)
                times[eng] = timeit(
                    lambda p=p, eng=eng: xr.open_dataset(p, engine=eng).load(), repeat
                )
            except Exception as e:  # noqa: BLE001
                times[eng] = f"n/a ({type(e).__name__})"
        times["annoray native"] = timeit(lambda p=p: ar.open_dataset(p).load(), repeat)
        rows.append((name, f"{size:.1f} MB", times))
    return rows


# ----- workflows -----


def workflows():
    air_x = xr.tutorial.open_dataset("air_temperature").load()
    air_r = ar.open_dataset(path("air_temperature")).load()
    sst_x = xr.tutorial.open_dataset("ersstv5").load()
    sst_r = ar.open_dataset(path("ersstv5")).load()
    rasm_x = xr.tutorial.open_dataset("rasm").load()
    rasm_r = ar.open_dataset(path("rasm")).load()
    era_x = xr.tutorial.open_dataset("eraint_uvz").load()
    era_r = ar.open_dataset(path("eraint_uvz")).load()
    roms_x = xr.tutorial.open_dataset("ROMS_example").load()
    roms_r = ar.open_dataset(path("ROMS_example")).load()

    # A larger air_temperature: 16 copies along time (~63 M values).
    n = 16
    big_x = xr.concat([air_x.air] * n, "time")
    big_x = big_x.assign_coords(
        time=np.arange(big_x.sizes["time"]) * np.timedelta64(6, "h") + np.datetime64("1900-01-01")
    )
    big_r = ar.from_xarray(big_x)

    a_x, a_r = air_x.air, air_r.air
    s_x, s_r = sst_x.sst, sst_r.sst
    clim_x = a_x.groupby("time.month").mean()
    clim_r = a_r.groupby("time.month").mean()
    sclim_x = s_x.groupby("time.month").mean()
    sclim_r = s_r.groupby("time.month").mean()
    bclim_x = big_x.groupby("time.month").mean()
    bclim_r = big_r.groupby("time.month").mean()

    # (dataset, operation, xarray, annoray)
    return [
        ("air_temperature", "mean over time", lambda: a_x.mean("time"), lambda: a_r.mean("time")),
        ("air_temperature", "std over time", lambda: a_x.std("time"), lambda: a_r.std("time")),
        (
            "air_temperature",
            "median over time",
            lambda: a_x.median("time"),
            lambda: a_r.median("time"),
        ),
        (
            "air_temperature",
            "monthly climatology",
            lambda: a_x.groupby("time.month").mean(),
            lambda: a_r.groupby("time.month").mean(),
        ),
        (
            "air_temperature",
            "anomaly from climatology",
            lambda: a_x.groupby("time.month") - clim_x,
            lambda: a_r.groupby("time.month") - clim_r,
        ),
        (
            "air_temperature",
            "daily means (resample 1D)",
            lambda: a_x.resample(time="1D").mean(),
            lambda: a_r.resample(time="1D").mean(),
        ),
        (
            "air_temperature",
            "rolling 8-step mean",
            lambda: a_x.rolling(time=8).mean(),
            lambda: a_r.rolling(time=8).mean(),
        ),
        ("air_temperature", "K → °C", lambda: a_x - 273.15, lambda: a_r.to("degC")),
        ("air_temperature", "a * 2 + a", lambda: a_x * 2.0 + a_x, lambda: a_r * 2.0 + a_r),
        (
            "air_temperature",
            "sel box + month",
            lambda: a_x.sel(lat=slice(60, 40), lon=slice(250, 280), time="2013-07"),
            lambda: a_r.sel(lat=slice(60, 40), lon=slice(250, 280), time="2013-07"),
        ),
        (
            "air_temperature",
            "interp to 1° lat",
            lambda: a_x.interp(lat=np.arange(20.0, 70.0, 1.0)),
            lambda: a_r.interp(lat=np.arange(20.0, 70.0, 1.0)),
        ),
        ("ersstv5", "mean over time", lambda: s_x.mean("time"), lambda: s_r.mean("time")),
        (
            "ersstv5",
            "monthly climatology",
            lambda: s_x.groupby("time.month").mean(),
            lambda: s_r.groupby("time.month").mean(),
        ),
        (
            "ersstv5",
            "anomaly from climatology",
            lambda: s_x.groupby("time.month") - sclim_x,
            lambda: s_r.groupby("time.month") - sclim_r,
        ),
        (
            "ersstv5",
            "annual max (resample YS)",
            lambda: s_x.resample(time="YS").max(),
            lambda: s_r.resample(time="YS").max(),
        ),
        (
            "rasm (noleap)",
            "mean over time",
            lambda: rasm_x.Tair.mean("time"),
            lambda: rasm_r.Tair.mean("time"),
        ),
        (
            "rasm (noleap)",
            "seasonal means (resample QS-DEC)",
            lambda: rasm_x.Tair.resample(time="QS-DEC").mean(),
            lambda: rasm_r.Tair.resample(time="QS-DEC").mean(),
        ),
        (
            "eraint_uvz",
            "u² + v² (level mean)",
            lambda: (era_x.u * era_x.u + era_x.v * era_x.v).mean("level"),
            lambda: (era_r.u * era_r.u + era_r.v * era_r.v).mean("level"),
        ),
        (
            "eraint_uvz",
            "zonal mean of z",
            lambda: era_x.z.mean("longitude"),
            lambda: era_r.z.mean("longitude"),
        ),
        (
            "ROMS_example",
            "depth mean of salt",
            lambda: roms_x.salt.mean("s_rho"),
            lambda: roms_r.salt.mean("s_rho"),
        ),
        (
            "air ×16 (63 M)",
            "mean over time",
            lambda: big_x.mean("time"),
            lambda: big_r.mean("time"),
        ),
        (
            "air ×16 (63 M)",
            "monthly climatology",
            lambda: big_x.groupby("time.month").mean(),
            lambda: big_r.groupby("time.month").mean(),
        ),
        (
            "air ×16 (63 M)",
            "anomaly from climatology",
            lambda: big_x.groupby("time.month") - bclim_x,
            lambda: big_r.groupby("time.month") - bclim_r,
        ),
        (
            "air ×16 (63 M)",
            "rolling 8-step mean",
            lambda: big_x.rolling(time=8).mean(),
            lambda: big_r.rolling(time=8).mean(),
        ),
        ("air ×16 (63 M)", "a * 2 + a", lambda: big_x * 2.0 + big_x, lambda: big_r * 2.0 + big_r),
        ("air ×16 (63 M)", "std over all", lambda: big_x.std(), lambda: big_r.std()),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {
        "python": sys.version.split()[0],
        "xarray": xr.__version__,
        "numpy": np.__version__,
        "annoray": ar.__version__,
        "cpus": os.cpu_count(),
        "io": [],
        "workflows": [],
    }

    names = ["air_temperature", "ersstv5", "rasm", "eraint_uvz", "ROMS_example", "basin_mask"]
    print(
        f"xarray {xr.__version__}, numpy {np.__version__}, annoray {ar.__version__}, {os.cpu_count()} CPUs"
    )
    print("\n## Open + load (median seconds)\n")
    print("| dataset | size | netcdf4 | h5netcdf | xarray engine=annoray | annoray native |")
    print("|---|---|---|---|---|---|")
    for name, size, t in io_rows(names, args.repeat):
        fmt = lambda v: f"{v * 1000:.1f} ms" if isinstance(v, float) else v
        print(
            f"| {name} | {size} | "
            + " | ".join(fmt(t[k]) for k in ["netcdf4", "h5netcdf", "annoray", "annoray native"])
            + " |"
        )
        out["io"].append({"dataset": name, "size": size, **t})

    print("\n## Workflows (median ms; results checked against xarray)\n")
    print("| dataset | operation | xarray | annoray | speedup |")
    print("|---|---|---|---|---|")
    for ds, op, fx, fr in workflows():
        try:
            same(fr(), fx(), f"{ds}: {op}")
        except Exception as e:  # noqa: BLE001
            print(
                f"| {ds} | {op} | | | **mismatch/error: {type(e).__name__}: {str(e).splitlines()[0][:120]}** |"
            )
            out["workflows"].append({"dataset": ds, "op": op, "error": str(e)[:300]})
            continue
        tx, tr = timeit(fx, args.repeat), timeit(fr, args.repeat)
        print(f"| {ds} | {op} | {tx * 1000:.2f} | {tr * 1000:.2f} | {tx / tr:.2f}× |")
        out["workflows"].append({"dataset": ds, "op": op, "xarray": tx, "annoray": tr})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
