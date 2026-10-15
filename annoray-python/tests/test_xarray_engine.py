"""annoray against xarray's own engines: reading must match netcdf4 exactly, and files written
by annoray must read back (with netcdf4) as the dataset written."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import annoray

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nc"
NC_FILES = ["xr_nc3.nc", "xr_nc4.nc", "raw_nc4.nc", "nat.nc"]
FORMATS = ["NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC", "NETCDF3_64BIT", "NETCDF3_64BIT_DATA"]


def _load_both(path, **kw):
    a = xr.open_dataset(path, engine="annoray", **kw)
    b = xr.open_dataset(path, engine="netcdf4", **kw)
    return a, b


def _assert_vars_identical(a: xr.Dataset, b: xr.Dataset):
    """Variable by variable; a variable xarray cannot decode must fail the same way."""
    assert list(a.variables) == list(b.variables)
    xr.testing.assert_identical(xr.Dataset(attrs=a.attrs), xr.Dataset(attrs=b.attrs))
    for k in b.variables:
        try:
            vb = b[k].load()
        except Exception as e:  # noqa: BLE001 - xarray's own decoding failures
            with pytest.raises(type(e)):
                a[k].load()
            continue
        xr.testing.assert_identical(a[k].load(), vb)


@pytest.mark.parametrize("name", NC_FILES)
@pytest.mark.parametrize("decode", [True, False])
def test_read_matches_netcdf4(name, decode):
    a, b = _load_both(FIXTURES / name, decode_cf=decode)
    _assert_vars_identical(a, b)
    assert a.encoding.get("unlimited_dims", set()) == b.encoding.get("unlimited_dims", set())


def test_encoding_matches_netcdf4():
    a, b = _load_both(FIXTURES / "xr_nc4.nc")
    for k in b.variables:
        for key in ("dtype", "chunksizes", "contiguous", "zlib", "complevel", "shuffle"):
            if key in b[k].encoding:
                assert a[k].encoding.get(key) == b[k].encoding[key], (k, key)
        for key in ("units", "calendar", "_FillValue", "scale_factor", "add_offset"):
            if key in b[k].encoding:
                np.testing.assert_equal(a[k].encoding.get(key), b[k].encoding[key])


def test_datatree_matches_netcdf4():
    a = xr.open_datatree(FIXTURES / "tree.nc", engine="annoray")
    b = xr.open_datatree(FIXTURES / "tree.nc", engine="netcdf4")
    xr.testing.assert_identical(a.load(), b.load())
    # One group at a time. (The reference here is h5netcdf: re-opening a file at a second
    # group with engine="netcdf4" segfaults in netCDF4-python, xarray 2026.9.)
    for group in b.children:
        a = xr.open_dataset(FIXTURES / "tree.nc", engine="annoray", group=group)
        _assert_vars_identical(a, xr.open_dataset(FIXTURES / "tree.nc", engine="h5netcdf", group=group))


def test_engine_is_registered_and_guessed():
    assert "annoray" in xr.backends.list_engines()
    from annoray.backend import AnnorayBackendEntrypoint

    entry = AnnorayBackendEntrypoint()
    assert entry.guess_can_open(FIXTURES / "xr_nc3.nc")
    assert entry.guess_can_open(FIXTURES / "xr_nc4.nc")
    assert not entry.guess_can_open(Path(__file__))


SELECTIONS = [
    {"time": 0},
    {"time": -1},
    {"time": slice(1, 3)},
    {"time": slice(None, None, -1)},
    {"time": slice(3, 0, -2), "lat": 1},
    {"lat": slice(0, 0)},
    {"time": [0, 2, 3]},
    {"time": [3, 1], "lon": [1, 0]},
    {"lat": np.array([True, False, True])},
]


@pytest.mark.parametrize("sel", SELECTIONS, ids=str)
def test_lazy_selections_match(sel):
    a, b = _load_both(FIXTURES / "xr_nc4.nc")
    _assert_vars_identical(a.isel(sel), b.isel(sel))


def test_dask_chunks_and_threads():
    pytest.importorskip("dask")
    a, b = _load_both(FIXTURES / "xr_nc4.nc", chunks={"time": 1})
    xr.testing.assert_identical(a.compute(), b.compute())


def test_parallel_reads_release_the_gil():
    a, b = _load_both(FIXTURES / "xr_nc4.nc")
    name = next(iter(b.data_vars))
    expect = b[name].values
    sels = list(itertools.product(range(expect.shape[0]), range(expect.shape[1])))
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda ij: a[name].isel(time=ij[0], lat=ij[1]).values, sels * 20))
    for (i, j), g in zip(sels * 20, got, strict=True):
        np.testing.assert_array_equal(g, expect[i, j])


def test_missing_file_is_an_oserror(tmp_path):
    with pytest.raises(OSError):
        xr.open_dataset(tmp_path / "nope.nc", engine="annoray")


# ----- writing -----


def _sample() -> xr.Dataset:
    t = pd.date_range("2000-01-01", periods=4, freq="6h").values.copy()
    t[2] = np.datetime64("NaT")
    rng = np.random.default_rng(0)
    temp = rng.normal(280, 5, (4, 3)).astype("float32")
    temp[1, 1] = np.nan
    return xr.Dataset(
        {
            "temp": (("time", "x"), temp, {"units": "K", "long_name": "temperature"}),
            "count": (("time", "x"), np.arange(12, dtype="int32").reshape(4, 3)),
            "flag": (("x",), np.array([True, False, True])),
            "lag": (("time",), np.array([0, 1, "NaT", 3], dtype="timedelta64[h]").astype("m8[ns]")),
            "name": (("x",), np.array(["alpha", "be", "c"], dtype=object)),
        },
        coords={
            "time": ("time", t),
            "x": ("x", np.array([10.0, 20.0, 30.0]), {"units": "m"}),
            "station": ("x", np.array(["a", "b", "c"], dtype=object)),
        },
        attrs={"title": "annoray test", "version": np.int32(3), "pair": np.array([1.5, 2.5])},
    )


def _roundtrip(ds, tmp_path, fmt="NETCDF4", **kw) -> xr.Dataset:
    path = tmp_path / f"out-{fmt}.nc"
    annoray.to_netcdf(ds, path, format=fmt, **kw)
    return xr.open_dataset(path, engine="netcdf4").load()


@pytest.mark.parametrize("fmt", FORMATS)
def test_write_reads_back_with_netcdf4(tmp_path, fmt):
    ds = _sample()
    if fmt.startswith("NETCDF3") or fmt == "NETCDF4_CLASSIC":
        ds = ds.drop_vars(["name", "station"])  # variable-length strings need netCDF-4
    back = _roundtrip(ds, tmp_path, fmt)
    xr.testing.assert_identical(back, ds)


def test_write_matches_xarray_output(tmp_path):
    ds = _sample()
    annoray.to_netcdf(ds, tmp_path / "r.nc")
    ds.to_netcdf(tmp_path / "x.nc", engine="netcdf4")
    a = xr.open_dataset(tmp_path / "r.nc", engine="netcdf4").load()
    b = xr.open_dataset(tmp_path / "x.nc", engine="netcdf4").load()
    xr.testing.assert_identical(a, b)


def test_cftime_roundtrip(tmp_path):
    dates = xr.date_range("2001-02-27", periods=4, freq="D", calendar="noleap", use_cftime=True)
    ds = xr.Dataset({"v": ("time", np.arange(4.0))}, coords={"time": dates})
    back = _roundtrip(ds, tmp_path)
    assert isinstance(back.time.values[0], cftime.DatetimeNoLeap)
    xr.testing.assert_identical(back, ds)


def test_packing_and_fill_encoding(tmp_path):
    ds = _sample()[["temp"]]
    enc = {"temp": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 280.0, "_FillValue": -32767}}
    path = tmp_path / "packed.nc"
    annoray.to_netcdf(ds, path, encoding=enc)
    raw = xr.open_dataset(path, engine="netcdf4", decode_cf=False)
    assert raw.temp.dtype == np.int16
    assert raw.temp.attrs["_FillValue"] == -32767
    back = xr.open_dataset(path, engine="netcdf4").load()
    np.testing.assert_allclose(back.temp, ds.temp, atol=0.005)
    assert np.isnan(back.temp.values[1, 1])


def test_time_encoding_is_honoured(tmp_path):
    ds = _sample()
    ds.time.encoding.update(units="hours since 1999-12-31", dtype="int64")
    path = tmp_path / "t.nc"
    annoray.to_netcdf(ds, path)
    raw = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    assert raw.time.attrs["units"] == "hours since 1999-12-31"
    assert raw.time.values[0] == 24


def test_unlimited_dims(tmp_path):
    ds = _sample()
    annoray.to_netcdf(ds, tmp_path / "u.nc", unlimited_dims=["time"])
    back = xr.open_dataset(tmp_path / "u.nc", engine="netcdf4")
    assert back.encoding["unlimited_dims"] == {"time"}


@pytest.mark.parametrize("codec", [{"zlib": True, "complevel": 5}, {"compression": "zlib"}])
def test_deflate_readable_by_netcdf4(tmp_path, codec):
    ds = xr.Dataset({"v": (("y", "x"), np.arange(4096.0).reshape(64, 64))})
    path = tmp_path / "z.nc"
    annoray.to_netcdf(ds, path, encoding={"v": {**codec, "chunksizes": (16, 32), "shuffle": True}})
    back = xr.open_dataset(path, engine="netcdf4")
    assert back.v.encoding["zlib"] and back.v.encoding["chunksizes"] == (16, 32)
    xr.testing.assert_identical(back.load(), ds)


@pytest.mark.parametrize("codec", ["zstd", "lz4", "blosc_lz4", "blosc_zstd"])
def test_other_codecs_roundtrip(tmp_path, codec):
    ds = xr.Dataset({"v": (("y", "x"), np.arange(4096.0).reshape(64, 64))})
    path = tmp_path / f"{codec}.nc"
    annoray.to_netcdf(ds, path, encoding={"v": {"compression": codec, "chunksizes": (32, 32)}})
    back = xr.open_dataset(path, engine="annoray")
    assert back.v.encoding["compression"] == codec
    xr.testing.assert_identical(back.load(), ds)


def test_annoray_reads_xarray_written_files(tmp_path):
    ds = _sample()
    for fmt in FORMATS:
        sub = ds if fmt == "NETCDF4" else ds.drop_vars(["name", "station"])
        path = tmp_path / f"x-{fmt}.nc"
        sub.to_netcdf(path, format=fmt, engine="netcdf4")
        xr.testing.assert_identical(xr.open_dataset(path, engine="annoray").load(), sub)


def test_unwritable_values_are_rejected(tmp_path):
    ds = xr.Dataset({"v": ("x", np.array([1 + 2j]))})
    with pytest.raises(TypeError):
        annoray.to_netcdf(ds, tmp_path / "c.nc")
    with pytest.raises(OSError):
        annoray.to_netcdf(_sample(), tmp_path / "missing-dir" / "x.nc")
