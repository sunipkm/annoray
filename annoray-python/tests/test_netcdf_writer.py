"""Writing into netCDF-4 files in place: streaming a new file from a template, appending
along an unlimited dimension (also to files netCDF-C wrote), and writing regions."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")

LAT = np.linspace(-80, 80, 5)
LON = np.arange(0, 360, 60.0)


def template(times=0):
    return ar.Dataset(
        {"temp": (("time", "lat", "lon"), np.zeros((times, 5, 6), "f4"), {"units": "K"})},
        coords={
            "time": ("time", np.array([], "datetime64[ns]")),
            "lat": ("lat", LAT, {"units": "degrees_north"}),
            "lon": ("lon", LON),
        },
    )


def step(k):
    t = np.array(["2000-01-01T00"], "datetime64[ns]") + np.timedelta64(k, "h")
    return ar.Dataset(
        {"temp": (("time", "lat", "lon"), np.full((1, 5, 6), 280 + k, "f4"), {"units": "K"})},
        coords={"time": ("time", t)},
    )


def test_stream_a_new_file(tmp_path):
    path = tmp_path / "s.nc"
    enc = {
        "temp": {"zlib": True, "chunksizes": (1, 5, 6)},
        "time": {"units": "hours since 2000-01-01"},
    }
    with ar.NetcdfWriter.create(path, template(), unlimited_dims=["time"], encoding=enc) as w:
        assert w.dim_len("time") == 0
        for k in range(4):
            w.append(step(k), "time")
            # The file so far, readable while it is written.
            now = w.dataset()
            assert now["temp"].shape == (k + 1, 5, 6)
            assert float(now["temp"].values[k, 0, 0]) == 280 + k
    with nc.Dataset(path) as f:
        assert f.dimensions["time"].isunlimited() and len(f.dimensions["time"]) == 4
        assert f["temp"][:, 2, 3].tolist() == [280, 281, 282, 283]
        assert f["time"][:].tolist() == [0, 1, 2, 3] and f["time"].units == "hours since 2000-01-01"
        assert f["lat"][:].tolist() == LAT.tolist()
        assert f["temp"].filters()["zlib"]
    ds = ar.open_dataset(path)
    assert str(ds["temp"].units) == "K"
    assert ds["time"].values[-1] == np.datetime64("2000-01-01T03")


def test_template_data_variables_start_as_fill(tmp_path):
    path = tmp_path / "t.nc"
    fixed = ar.Dataset(
        {"v": (("y", "x"), np.ones((4, 6)))},
        coords={"x": ("x", np.arange(6.0)), "y": ("y", np.arange(4.0))},
    )
    with ar.NetcdfWriter.create(path, fixed) as w:
        assert np.isnan(w.dataset()["v"].values).all()
        tile = ar.Dataset({"v": (("y", "x"), np.full((2, 3), 7.0))})
        w.write(tile, region={"y": 2, "x": slice(3, 6)})
    v = ar.open_dataset(path)["v"].values
    assert (v[2:, 3:] == 7).all() and np.isnan(v[:2]).all() and np.isnan(v[2:, :3]).all()


def write_netcdf_c(path):
    with nc.Dataset(path, "w") as f:
        f.createDimension("time", None)
        f.createDimension("x", 4)
        f.createDimension("nchar", 5)
        t = f.createVariable("time", "f8", ("time",))
        t.units = "days since 2000-01-01"
        v = f.createVariable("v", "i2", ("time", "x"), zlib=True)
        v.scale_factor = 0.5
        v.add_offset = 10.0
        v.units = "m"
        v.missing_value = np.int16(-999)
        s = f.createVariable("name", "S1", ("time", "nchar"))
        s.set_auto_chartostring(False)
        c = f.createVariable("c", "f4", ("x",))
        c[:] = [1, 2, 3, 4]
        t[0:2] = [0, 1]
        v[0:2] = np.array([[10.5, 11, 11.5, 12], [13, 14, 15, 16]])
        s[0:2] = np.array([[b"a", b"b", b"", b"", b""], [b"c", b"d", b"e", b"", b""]], "S1")


def test_append_to_a_netcdf_c_file(tmp_path):
    path = tmp_path / "c.nc"
    write_netcdf_c(path)
    part = ar.Dataset(
        {
            # In km: converted to the file's metres, then packed with its scale and offset.
            "v": (("time", "x"), np.array([[1.0, 2.0, 3.0, np.nan]]), {"units": "km"}),
            "name": ("time", np.array(["xyz"])),
        },
        coords={"time": ("time", np.array(["2000-01-03"], "datetime64[ns]"))},
    )
    with ar.NetcdfWriter(path) as w:
        w.append(part, "time")
    with nc.Dataset(path) as f:
        assert len(f.dimensions["time"]) == 3 and f["time"][:].tolist() == [0, 1, 2]
        f.set_auto_mask(False)
        f.set_auto_scale(False)
        assert f["v"][2].tolist() == [1980, 3980, 5980, -999]
    ds = ar.open_dataset(path)
    np.testing.assert_array_equal(ds["v"].values[2], [1000, 2000, 3000, np.nan])
    assert ds["name"].values.tolist() == ["ab", "cde", "xyz"]


def test_region_overwrites_strings_whole(tmp_path):
    path = tmp_path / "c.nc"
    write_netcdf_c(path)
    with ar.NetcdfWriter(path) as w:
        w.write(ar.Dataset({"name": ("time", np.array(["q"]))}), region={"time": 1})
        w.write(ar.Dataset({"c": ("x", np.array([9.0, 8.0], "f4"))}), region={"x": 1})
    ds = ar.open_dataset(path)
    assert ds["name"].values.tolist() == ["ab", "q"]
    assert ds["c"].values.tolist() == [1, 9, 8, 4]


def test_to_netcdf_append_dim_and_region(tmp_path):
    path = tmp_path / "s.nc"
    first = step(0)
    first.to_netcdf(
        path, unlimited_dims=["time"], encoding={"time": {"units": "hours since 2000-01-01"}}
    )
    step(1).to_netcdf(path, mode="a", append_dim="time")
    step(2).to_netcdf(path, mode="a", append_dim="time")
    assert ar.open_dataset(path)["temp"].values[:, 0, 0].tolist() == [280, 281, 282]
    fix = ar.Dataset({"temp": (("time", "lat", "lon"), np.zeros((1, 5, 6), "f4"), {"units": "K"})})
    fix.to_netcdf(path, mode="a", region={"time": 1})
    assert ar.open_dataset(path)["temp"].values[:, 0, 0].tolist() == [280, 0, 282]
    with pytest.raises(ValueError, match="mode"):
        step(3).to_netcdf(path, append_dim="time")


def test_times_that_do_not_fit_the_files_units(tmp_path):
    path = tmp_path / "s.nc"
    step(0).to_netcdf(path, unlimited_dims=["time"])  # one time: "days since …", int64
    with pytest.raises(ValueError, match="cannot be stored as int64"):
        step(1).to_netcdf(path, mode="a", append_dim="time")
    assert ar.open_dataset(path).sizes["time"] == 1


def test_errors(tmp_path):
    path = tmp_path / "c.nc"
    write_netcdf_c(path)
    w = ar.NetcdfWriter(path)
    no_name = ar.Dataset(
        {"v": (("time", "x"), np.zeros((1, 4)), {"units": "m"})},
        coords={"time": ("time", np.array(["2000-01-03"], "datetime64[ns]"))},
    )
    with pytest.raises(OSError, match="missing"):
        w.append(no_name, "time")
    with pytest.raises(OSError, match="not unlimited"):
        w.append(ar.Dataset({"c": ("x", np.zeros(4, "f4"))}), "x")
    with pytest.raises(Exception, match="(?i)unit"):
        w.write(ar.Dataset({"c": ("x", np.zeros(2, "f4"), {"units": "s"})}), region={"x": 0})
    with pytest.raises(OSError, match="not in the file"):
        w.write(ar.Dataset({"nope": ("x", np.zeros(4))}))
    with pytest.raises(OSError, match="outside"):
        w.write(ar.Dataset({"c": ("x", np.zeros(3, "f4"))}), region={"x": 2})
    w.close()
    assert w.closed
    with pytest.raises(ValueError, match="closed"):
        w.dim_len("time")


def test_hdf5_keeps_appending_after_us(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "s.nc"
    with ar.NetcdfWriter.create(path, template(), unlimited_dims=["time"]) as w:
        w.append(step(0), "time")
    with h5py.File(path, "a") as f:
        t = f["temp"]
        t.resize((3, 5, 6))
        t[1:] = 5.0
        f["time"].resize((3,))
        f["time"][1:] = [1e9, 2e9]
    assert ar.open_dataset(path)["temp"].values[:, 0, 0].tolist() == [280, 5, 5]


def test_newer_hdf5_indexes_are_refused(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "l.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("v", data=np.zeros((2, 3)), maxshape=(None, 3), chunks=(1, 3))
    with pytest.raises(OSError, match="extensible array"):
        ar.NetcdfWriter(path).write(
            ar.Dataset({"v": (("phony_dim_0", "phony_dim_1"), np.ones((1, 3)))})
        )
