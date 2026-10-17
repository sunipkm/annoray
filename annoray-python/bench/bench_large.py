"""Benchmarks on large files: annoray against xarray (with bottleneck, flox, numbagg and
dask installed, as a performance-minded user has them).

A year of hourly 2 m temperature on a 1° grid (time 8760 × lat 181 × lon 360, float32,
2.3 GB of values) is written three ways: netCDF-4 uncompressed, netCDF-4 zlib, and Zarr v3
(zstd). Every annoray result is checked against xarray's before timing. File reads are
with a warm page cache; peak memory is the peak anonymous memory (allocations, not mapped
file pages) of a fresh process, sampled while the timed repeats run.

    python bench_large.py --generate            # write the files (once)
    python bench_large.py [--repeat N] [--json out.json] [--only io|compute]
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DATA = Path(os.environ.get("BENCH_DATA", Path.home() / "bench" / "data"))
NT, NLAT, NLON = 8760, 181, 360
FILES = {
    "nc4": DATA / "t2m.nc",
    "nc4-zlib": DATA / "t2m_zlib.nc",
    "zarr": DATA / "t2m.zarr",
}


# ----- data -----


def field(t0: int, n: int) -> np.ndarray:
    """Smooth, deterministic temperatures (K) for hours ``t0 .. t0 + n``."""
    lat = np.deg2rad(np.linspace(-90, 90, NLAT))[None, :, None]
    lon = np.deg2rad(np.arange(NLON))[None, None, :]
    t = np.arange(t0, t0 + n)[:, None, None]
    season = 15 * np.sin(2 * np.pi * t / NT) * np.sin(lat)
    diurnal = 5 * np.sin(2 * np.pi * (t / 24 + lon / (2 * np.pi)))
    rng = np.random.default_rng(t0)
    noise = rng.normal(0, 0.5, (n, NLAT, NLON))
    return (250 + 40 * np.cos(lat) + season + diurnal * np.cos(lat) + noise).astype(np.float32)


def generate() -> None:
    import netCDF4
    import xarray as xr

    DATA.mkdir(parents=True, exist_ok=True)
    for key, zlib in (("nc4", False), ("nc4-zlib", True)):
        path = FILES[key]
        if path.exists():
            continue
        with netCDF4.Dataset(path, "w") as d:
            d.createDimension("time", NT)
            d.createDimension("lat", NLAT)
            d.createDimension("lon", NLON)
            t = d.createVariable("time", "i8", ("time",))
            t.units, t.calendar = "hours since 2001-01-01", "standard"
            t[:] = np.arange(NT)
            la = d.createVariable("lat", "f8", ("lat",))
            la.units = "degrees_north"
            la[:] = np.linspace(-90, 90, NLAT)
            lo = d.createVariable("lon", "f8", ("lon",))
            lo.units = "degrees_east"
            lo[:] = np.arange(NLON, dtype=float)
            v = d.createVariable(
                "t2m", "f4", ("time", "lat", "lon"), chunksizes=(24, NLAT, NLON),
                zlib=zlib, complevel=1, shuffle=zlib,
            )
            v.units, v.long_name = "K", "2 m temperature"
            for t0 in range(0, NT, 720):
                n = min(720, NT - t0)
                v[t0 : t0 + n] = field(t0, n)
        print("wrote", path, f"{path.stat().st_size / 1e9:.2f} GB")
    if not FILES["zarr"].exists():
        ds = xr.open_dataset(FILES["nc4"], chunks={"time": 720})
        ds.to_zarr(FILES["zarr"], zarr_format=3, encoding={"t2m": {"chunks": (24, NLAT, NLON)}})
        size = sum(f.stat().st_size for f in FILES["zarr"].rglob("*") if f.is_file())
        print("wrote", FILES["zarr"], f"{size / 1e9:.2f} GB")


# ----- timing -----


def timeit(f, repeat: int) -> float:
    f()  # warm-up (page cache, imports)
    ts = []
    for _ in range(repeat):
        s = time.perf_counter()
        f()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


ISOLATED = r'''
import os, sys, time, json, resource, statistics, threading, warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np, xarray as xr, annoray as ar
P = json.loads(sys.argv[1])
# Peak anonymous memory (allocations), sampled: unlike ru_maxrss it leaves out the pages of
# memory-mapped files, which the operating system can reclaim. Linux only; elsewhere the
# resident-set peak.
def anon_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        return None
peak = [0.0]
done = [False]
def sample():
    while not done[0]:
        peak[0] = max(peak[0], anon_gb() or 0.0)
        time.sleep(0.002)
def run():
{body}
run()  # warm-up: page cache and imports
sampler = threading.Thread(target=sample, daemon=True)
sampler.start()
ts = []
for _ in range(P["repeat"]):
    out = None  # the previous result is released first: one result alive at a time
    s = time.perf_counter(); out = run(); ts.append(time.perf_counter() - s)
done[0] = True
sampler.join()
mem = peak[0] if anon_gb() is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(json.dumps({{"t": statistics.median(ts), "rss_gb": mem}}))
'''


def isolated(body: str, repeat: int, **params) -> dict:
    """Run ``body`` (the body of ``run()``, indented) in a fresh process: median time and
    the process's peak resident memory."""
    code = ISOLATED.format(body=body)
    p = {"repeat": repeat, **{k: str(v) for k, v in params.items()}}
    r = subprocess.run(
        [sys.executable, "-c", code, json.dumps(p)], capture_output=True, text=True, timeout=3600
    )
    if r.returncode:
        return {"error": r.stderr.strip().splitlines()[-1][:200]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def io_cases() -> list[tuple[str, str, dict[str, str]]]:
    """(file, operation, {library: body of run()})."""
    nc, zl, zr = (str(FILES[k]) for k in ("nc4", "nc4-zlib", "zarr"))
    load = {
        "xarray netcdf4": '    return xr.open_dataset(P["f"], engine="netcdf4").t2m.load()',
        "xarray h5netcdf": '    return xr.open_dataset(P["f"], engine="h5netcdf").t2m.load()',
        "annoray": '    return ar.open_dataset(P["f"]).t2m.load()',
    }
    stream = {
        "xarray (load, then mean)": '    return xr.open_dataset(P["f"]).t2m.mean("time").load()',
        "xarray + dask": '    return xr.open_dataset(P["f"], chunks={"time": 720}).t2m.mean("time").compute()',
        "annoray (lazy, streamed)": '    return ar.open_dataset(P["f"]).t2m.mean("time")',
    }
    point = {
        "xarray netcdf4": '    return xr.open_dataset(P["f"]).t2m.isel(lat=90, lon=180).load()',
        "annoray": '    return ar.open_dataset(P["f"]).t2m.isel(lat=90, lon=180).load()',
    }
    day = {
        "xarray netcdf4": '    return xr.open_dataset(P["f"]).t2m.sel(time="2001-07-01").load()',
        "annoray": '    return ar.open_dataset(P["f"]).t2m.sel(time="2001-07-01").load()',
    }
    zload = {
        "xarray zarr": '    return xr.open_zarr(P["f"]).t2m.load()',
        "annoray": '    return ar.open_dataset(P["f"]).t2m.load()',
    }
    zstream = {
        "xarray zarr (dask)": '    return xr.open_zarr(P["f"]).t2m.mean("time").compute()',
        "annoray (lazy, streamed)": '    return ar.open_dataset(P["f"]).t2m.mean("time")',
    }
    out_nc = str(DATA / "out.nc")
    write = {
        "xarray netcdf4": f'    ds = xr.open_dataset(P["f"]).load() if not hasattr(run, "x") else run.x\n    run.x = ds\n    ds.to_netcdf("{out_nc}")',
        "annoray": f'    ds = ar.open_dataset(P["f"]).load() if not hasattr(run, "a") else run.a\n    run.a = ds\n    ds.to_netcdf("{out_nc}")',
    }
    out_zarr = str(DATA / "out.zarr")
    zwrite = {
        "xarray zarr": f'    ds = xr.open_dataset(P["f"]).load() if not hasattr(run, "x") else run.x\n    run.x = ds\n    ds.to_zarr("{out_zarr}", mode="w", zarr_format=3)',
        "annoray": f'    ds = ar.open_dataset(P["f"]).load() if not hasattr(run, "a") else run.a\n    run.a = ds\n    ds.to_zarr("{out_zarr}", mode="w")',
    }
    return [
        (nc, "open + load all", load),
        (zl, "open + load all", load),
        (zr, "open + load all", zload),
        (nc, "mean over time (from disk)", stream),
        (zl, "mean over time (from disk)", stream),
        (zr, "mean over time (from disk)", zstream),
        (nc, "one point, all times", point),
        (zl, "one point, all times", point),
        (nc, "one day, whole grid", day),
        (nc, "write netCDF-4 (from memory)", write),
        (nc, "write Zarr v3 (from memory)", zwrite),
    ]


# ----- compute -----


def same(r, x, what: str) -> None:
    import xarray as xr

    got = r.to_xarray() if hasattr(r, "to_xarray") else r
    if isinstance(got, xr.DataArray) and isinstance(x, xr.DataArray):
        got = got.rename(x.name).drop_attrs()
        x = x.drop_attrs()
    f32 = getattr(x, "dtype", None) == np.float32
    try:
        xr.testing.assert_allclose(got, x, rtol=1e-5 if f32 else 1e-7, atol=1e-3 if f32 else 1e-8)
    except AssertionError as e:
        raise AssertionError(f"{what}: results differ\n{str(e)[:400]}") from None


def compute_cases():
    import xarray as xr

    import annoray as ar

    x = xr.open_dataset(FILES["nc4"]).t2m.load()
    a = ar.open_dataset(FILES["nc4"]).t2m.load()
    K = ar.Quantity(273.15, "K")
    clim_x = x.groupby("time.month").mean()
    clim_a = a.groupby("time.month").mean()
    b_x, b_a = x.roll(time=1) * 0.5 + x * 0.5, a.roll(time=1) * 0.5 + a * 0.5
    return [
        ("mean over time", lambda: x.mean("time"), lambda: a.mean("time")),
        ("std over time", lambda: x.std("time"), lambda: a.std("time")),
        ("max over lat/lon", lambda: x.max(["lat", "lon"]), lambda: a.max(["lat", "lon"])),
        ("monthly climatology", lambda: x.groupby("time.month").mean(), lambda: a.groupby("time.month").mean()),
        ("anomaly from climatology", lambda: x.groupby("time.month") - clim_x, lambda: a.groupby("time.month") - clim_a),
        ("daily means (resample 1D)", lambda: x.resample(time="1D").mean(), lambda: a.resample(time="1D").mean()),
        ("rolling 24-h mean", lambda: x.rolling(time=24).mean(), lambda: a.rolling(time=24).mean()),
        ("rolling_exp mean (span 24)", lambda: x.rolling_exp(time=24).mean(), lambda: a.rolling_exp(time=24).mean()),
        ("90th percentile over time", lambda: x.quantile(0.9, "time"), lambda: a.quantile(0.9, "time")),
        ("K → °C", lambda: x - 273.15, lambda: a.to("degC")),
        ("a * 2 + a", lambda: x * 2.0 + x, lambda: a * 2.0 + a),
        ("where(a > 0 °C, a, 0 °C)", lambda: xr.where(x > 273.15, x, 273.15), lambda: ar.where(a > K, a, K)),
        ("corr over time", lambda: xr.corr(x, b_x, dim="time"), lambda: ar.corr(a, b_a, dim="time")),
        ("convert_calendar noleap", lambda: x.convert_calendar("noleap"), lambda: a.convert_calendar("noleap")),
        ("sel box + month", lambda: x.sel(lat=slice(30, 60), lon=slice(200, 300), time="2001-07"),
         lambda: a.sel(lat=slice(30, 60), lon=slice(200, 300), time="2001-07")),
        ("interp to 0.5° lat", lambda: x.isel(time=slice(0, 744)).interp(lat=np.arange(-90.0, 90.5, 0.5)),
         lambda: a.isel(time=slice(0, 744)).interp(lat=np.arange(-90.0, 90.5, 0.5))),
        ("astype float64", lambda: x.astype("float64"), lambda: a.astype("float64")),
        ("equals (copy)", lambda: x.equals(x.copy()), lambda: a.equals(a.copy())),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--only", choices=["io", "compute"])
    args = ap.parse_args()
    if args.generate:
        generate()
        return

    import xarray as xr

    import annoray as ar

    out: dict = {
        "python": sys.version.split()[0],
        "xarray": xr.__version__,
        "numpy": np.__version__,
        "annoray": ar.__version__,
        "cpus": os.cpu_count(),
        "io": [],
        "compute": [],
    }
    print(f"xarray {xr.__version__}, numpy {np.__version__}, annoray {ar.__version__}, {os.cpu_count()} CPUs")
    print(f"data: time {NT} × lat {NLAT} × lon {NLON} float32 ({NT * NLAT * NLON * 4 / 1e9:.1f} GB)")

    if args.only in (None, "io"):
        print("\n## Files (median s; peak anonymous memory of the process)\n")
        print("| file | operation | library | time | peak memory |")
        print("|---|---|---|---|---|")
        for f, op, libs in io_cases():
            for lib, body in libs.items():
                r = isolated(body, args.repeat, f=f)
                name = Path(f).name
                if "error" in r:
                    print(f"| {name} | {op} | {lib} | error: {r['error']} | |")
                else:
                    print(f"| {name} | {op} | {lib} | {r['t']:.3f} | {r['rss_gb']:.2f} GB |")
                out["io"].append({"file": name, "op": op, "lib": lib, **r})
                sys.stdout.flush()

    if args.only in (None, "compute"):
        print("\n## In memory (median s; results checked against xarray)\n")
        print("| operation | xarray | annoray | speedup |")
        print("|---|---|---|---|")
        for op, fx, fa in compute_cases():
            try:
                rx, ra = fx(), fa()
                if isinstance(rx, bool):
                    assert rx == ra, f"{op}: {ra} != {rx}"
                else:
                    same(ra, rx, op)
            except Exception as e:  # noqa: BLE001
                print(f"| {op} | | | **error: {type(e).__name__}: {str(e).splitlines()[0][:100]}** |")
                out["compute"].append({"op": op, "error": str(e)[:300]})
                continue
            tx, ta = timeit(fx, args.repeat), timeit(fa, args.repeat)
            print(f"| {op} | {tx:.3f} | {ta:.3f} | {tx / ta:.2f}× |")
            out["compute"].append({"op": op, "xarray": tx, "annoray": ta})
            sys.stdout.flush()

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
