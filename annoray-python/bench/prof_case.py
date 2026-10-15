"""Run one of the slow benchmark cases in a loop, for a profiler.

    py-spy record --native -r 500 -f raw -o out.txt -- python prof_case.py CASE SECONDS
"""

from __future__ import annotations

import os

# scipy (imported by xarray.tutorial) starts OpenBLAS threads that spin idle and take cores
# from both libraries; one is enough here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import warnings

import numpy as np
import xarray as xr

import annoray as ar

warnings.filterwarnings("ignore")


def path(name):
    return str(xr.tutorial.open_dataset(name).encoding["source"])


def setup(case):
    if case == "anomaly":
        a = ar.open_dataset(path("air_temperature")).load().air
        clim = a.groupby("time.month").mean()
        return lambda: a.groupby("time.month") - clim
    if case in ("anomaly_big", "arith_big"):
        x = xr.tutorial.open_dataset("air_temperature").load().air
        x = xr.concat([x] * 16, "time")
        x = x.assign_coords(
            time=np.arange(x.sizes["time"]) * np.timedelta64(6, "h") + np.datetime64("1900-01-01")
        )
        a = ar.from_xarray(x)
        if case == "arith_big":
            return lambda: a * 2.0 + a
        clim = a.groupby("time.month").mean()
        return lambda: a.groupby("time.month") - clim
    if case == "resample_ys":
        s = ar.open_dataset(path("ersstv5")).load().sst
        return lambda: s.resample(time="YS").max()
    if case == "to_degc":
        a = ar.open_dataset(path("air_temperature")).load().air
        return lambda: a.to("degC")
    if case == "f32_arith":
        e = ar.open_dataset(path("eraint_uvz")).load()
        u, v = e.u, e.v
        return lambda: u * u + v * v
    if case == "middle_mean":
        s = ar.open_dataset(path("ROMS_example")).load().salt
        return lambda: s.mean("s_rho")
    if case == "open_air":
        p = path("air_temperature")
        return lambda: ar.open_dataset(p).load()
    if case == "open_sst":
        p = path("ersstv5")
        return lambda: ar.open_dataset(p).load()
    raise SystemExit(f"unknown case {case}")


def main():
    case, seconds = sys.argv[1], float(sys.argv[2])
    f = setup(case)
    f()
    n, end = 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        f()
        n += 1
    print(f"{case}: {n} runs, {seconds / n * 1000:.2f} ms each", file=sys.stderr)


if __name__ == "__main__":
    main()
