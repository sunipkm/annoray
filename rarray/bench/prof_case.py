"""Run one of the slow benchmark cases in a loop, for a profiler.

    py-spy record --native -r 500 -f raw -o out.txt -- python prof_case.py CASE SECONDS
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import xarray as xr

import rarray as ra

warnings.filterwarnings("ignore")


def path(name):
    return str(xr.tutorial.open_dataset(name).encoding["source"])


def setup(case):
    if case == "anomaly":
        a = ra.open_dataset(path("air_temperature")).load().air
        clim = a.groupby("time.month").mean()
        return lambda: a.groupby("time.month") - clim
    if case == "resample_ys":
        s = ra.open_dataset(path("ersstv5")).load().sst
        return lambda: s.resample(time="YS").max()
    if case == "to_degc":
        a = ra.open_dataset(path("air_temperature")).load().air
        return lambda: a.to("degC")
    if case == "f32_arith":
        e = ra.open_dataset(path("eraint_uvz")).load()
        u, v = e.u, e.v
        return lambda: u * u + v * v
    if case == "middle_mean":
        s = ra.open_dataset(path("ROMS_example")).load().salt
        return lambda: s.mean("s_rho")
    if case == "open_air":
        p = path("air_temperature")
        return lambda: ra.open_dataset(p).load()
    if case == "open_sst":
        p = path("ersstv5")
        return lambda: ra.open_dataset(p).load()
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
