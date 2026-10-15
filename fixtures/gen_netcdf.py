"""Golden netCDF files for the I/O tests, written by Python xarray / netCDF4.

Run from the repository root:  python3 fixtures/gen_netcdf.py
Writes fixtures/nc/*.nc and prints what xarray decodes from them (the Rust tests hardcode it).
"""

import os
import warnings

import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
OUT = os.path.join(os.path.dirname(__file__), "nc")
os.makedirs(OUT, exist_ok=True)


def dataset():
    time = pd.date_range("2000-01-01", periods=4, freq="6h")
    lat = np.array([-10.0, 0.0, 10.0], dtype="float32")
    lon = np.array([100.0, 110.0])
    temp = np.arange(24, dtype="float32").reshape(4, 3, 2) + 270
    temp[1, 1, 1] = np.nan
    precip = np.round(np.linspace(0, 3, 24).reshape(4, 3, 2), 2)
    precip[0, 0, 0] = np.nan
    model_time = [cftime.datetime(2001, 2, d, calendar="noleap") for d in (1, 15, 28)]
    ds = xr.Dataset(
        {
            "temp": (("time", "lat", "lon"), temp, {"units": "K", "long_name": "air temperature"}),
            "precip": (("time", "lat", "lon"), precip, {"units": "mm/day"}),
            "station": (("lat",), np.array(["a", "bb", "ccc"], dtype=object)),
            "flag": (("lat",), np.array([1, 200, 255], dtype="uint8")),
            "is_land": (("lat",), np.array([True, False, True])),
            "model_time": (("lat",), model_time),
        },
        coords={
            "time": time,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "height": ((), 2.0, {"units": "m"}),
        },
        attrs={"title": "golden file", "version": np.int32(3)},
    )
    ds["precip"].encoding = {"dtype": "int16", "scale_factor": 0.01, "add_offset": 1.5, "_FillValue": -32767}
    ds["temp"].encoding = {"zlib": True, "complevel": 4, "chunksizes": (1, 3, 2)}
    return ds


def show(path):
    print(f"== {os.path.basename(path)}")
    ds = xr.open_dataset(path)
    for name in ["temp", "precip", "station", "flag", "is_land", "model_time", "time", "lat", "height"]:
        if name not in ds:
            continue
        v = ds[name]
        vals = v.values.ravel().tolist()
        if name in ("time",):
            vals = [str(x) for x in v.values]
        if name == "model_time":
            vals = [str(x) for x in v.values]
        print(f"  {name}: dims={v.dims} dtype={v.dtype} units={v.attrs.get('units')} values={vals}")
    print(f"  coords={sorted(map(str, ds.coords))} data_vars={sorted(map(str, ds.data_vars))}")
    ds.close()


def raw_file(path):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("x", 3)
        nc.createDimension("nchar", 5)
        nc.setncattr("ints", np.array([1, 2, 3], dtype="int32"))
        b = nc.createVariable("ubyte_as_signed", "i1", ("x",))
        b.set_auto_maskandscale(False)
        b.setncattr("_Unsigned", "true")
        b[:] = np.array([-1, 0, 127], dtype="i1")
        c = nc.createVariable("names", "S1", ("x", "nchar"))
        c.set_auto_chartostring(False)
        chars = [[ch.encode() for ch in w.ljust(5, "\0")] for w in ["one", "three", "x"]]
        c[:] = np.array(chars, dtype="S1")
        m = nc.createVariable("with_missing", "f8", ("x",))
        m.set_auto_maskandscale(False)
        m.missing_value = -1.0
        m[:] = [1.0, -1.0, 3.0]
        g = nc.createGroup("grp")
        g.createDimension("y", 2)
        gv = g.createVariable("gvar", "f4", ("x", "y"))
        gv.units = "km"
        gv[:] = np.arange(6, dtype="f4").reshape(3, 2)
        sub = g.createGroup("deeper")
        sub.title = "nested"


def main():
    print(f"# xarray {xr.__version__}, netCDF4 {netCDF4.__version__}, netCDF-C {netCDF4.__netcdf4libversion__}")
    ds = dataset()
    p4 = os.path.join(OUT, "xr_nc4.nc")
    ds.to_netcdf(p4, format="NETCDF4", unlimited_dims=["time"])
    show(p4)
    p3 = os.path.join(OUT, "xr_nc3.nc")
    ds3 = ds.drop_vars("flag")  # xarray refuses uint8 in netCDF-3
    for v in ds3.variables.values():
        v.encoding = {k: x for k, x in v.encoding.items() if k not in ("zlib", "complevel", "chunksizes")}
    ds3.to_netcdf(p3, format="NETCDF3_64BIT")
    show(p3)
    pr = os.path.join(OUT, "raw_nc4.nc")
    raw_file(pr)
    print("== raw_nc4.nc (decoded by xarray)")
    r = xr.open_dataset(pr)
    for name in r.variables:
        print(f"  {name}: dtype={r[name].dtype} values={r[name].values.ravel().tolist()}")
    g = xr.open_dataset(pr, group="grp")
    print(f"  grp/gvar: units={g['gvar'].attrs.get('units')} values={g['gvar'].values.ravel().tolist()}")


if __name__ == "__main__":
    main()
