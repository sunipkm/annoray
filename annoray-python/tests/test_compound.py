"""Compound (record) variables: read as one variable per field (`var.field`, nested fields
dotted, array members with dimensions of their own), written back as one compound variable."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")
h5py = pytest.importorskip("h5py")

INNER = np.dtype([("a", "<f4"), ("b", "<i2", (3,))])
REC = np.dtype([("x", "<f8"), ("n", "<i4"), ("in", INNER)])


def records():
    d = np.zeros(4, REC)
    d["x"] = [1.5, 2, 3, 4]
    d["n"] = [1, 2, 3, 4]
    d["in"]["a"] = [0.5, 1, 2, 3]
    d["in"]["b"] = np.arange(12).reshape(4, 3)
    return d


def write_netcdf(path):
    with nc.Dataset(path, "w") as f:
        f.createCompoundType(INNER, "inner_t")
        t = f.createCompoundType(REC, "rec_t")
        f.createDimension("t", 4)
        v = f.createVariable("rec", t, ("t",))
        v[:] = records()
        v.units = "m"


def test_read_one_variable_per_field(tmp_path):
    write_netcdf(tmp_path / "c.nc")
    ds = ar.open_dataset(tmp_path / "c.nc")
    assert list(ds.data_vars) == ["rec.x", "rec.n", "rec.in.a", "rec.in.b"]
    assert ds["rec.x"].dims == ("t",) and ds["rec.x"].values.tolist() == [1.5, 2, 3, 4]
    assert ds["rec.n"].dtype == np.int32
    assert ds["rec.in.a"].dtype == np.float32
    assert ds["rec.in.b"].dims == ("t", "rec.in.b_dim0")
    assert ds["rec.in.b"].values.tolist() == np.arange(12).reshape(4, 3).tolist()
    assert str(ds["rec.x"].units) == "m"


def test_netcdf_round_trip(tmp_path):
    write_netcdf(tmp_path / "c.nc")
    ar.open_dataset(tmp_path / "c.nc").to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert list(f.variables) == ["rec"] and list(f.dimensions) == ["t"]
        v = f["rec"]
        assert v.units == "m"
        got = v[:]
    want = records()
    for field in ("x", "n"):
        np.testing.assert_array_equal(got[field], want[field])
    np.testing.assert_array_equal(got["in"]["a"], want["in"]["a"])
    np.testing.assert_array_equal(got["in"]["b"], want["in"]["b"])
    back = ar.open_dataset(tmp_path / "out.nc")
    assert list(back.data_vars) == ["rec.x", "rec.n", "rec.in.a", "rec.in.b"]


def test_h5py_records_round_trip(tmp_path):
    dt = np.dtype([("name", "S5"), ("count", ">i4"), ("label", h5py.string_dtype()), ("w", "<f2")])
    d = np.array([(b"ab", 7, "first", 0.5), (b"cdefg", -1, "second", 2.0)], dtype=dt)
    with h5py.File(tmp_path / "h.h5", "w") as f:
        f["r"] = d
    ds = ar.open_dataset(tmp_path / "h.h5")
    assert ds["r.name"].values.tolist() == ["ab", "cdefg"]
    assert ds["r.count"].values.tolist() == [7, -1]
    assert ds["r.label"].values.tolist() == ["first", "second"]
    assert ds["r.w"].values.tolist() == [0.5, 2.0]
    ds.to_netcdf(tmp_path / "out.nc")
    with h5py.File(tmp_path / "out.nc") as f:
        got = f["r"][...]
        assert f["r"].dtype["count"] == np.dtype(">i4")
        assert f["r"].dtype["w"] == np.dtype("<f2")
    assert got["name"].tolist() == [b"ab", b"cdefg"]
    assert got["count"].tolist() == [7, -1]
    assert [x.decode() for x in got["label"]] == ["first", "second"]
    assert got["w"].tolist() == [0.5, 2.0]


def test_missing_field_writes_separate_variables(tmp_path):
    write_netcdf(tmp_path / "c.nc")
    ds = ar.open_dataset(tmp_path / "c.nc").drop_vars("rec.n")
    ds.to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert "rec" not in f.variables
        assert {"rec.x", "rec.in.a", "rec.in.b"} <= set(f.variables)


def test_put_keeps_the_record(tmp_path):
    write_netcdf(tmp_path / "c.nc")
    ds = ar.open_dataset(tmp_path / "c.nc")
    ds = ds.assign({"rec.n": ds["rec.n"].put(ar.Quantity(40, "m"), t=0)})
    ds.to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert list(f.variables) == ["rec"]
        assert f["rec"][:]["n"].tolist() == [40, 2, 3, 4]


def test_new_dtype_writes_separate_variables(tmp_path):
    # A new dtype drops the encoding (as in xarray), the record type with it.
    write_netcdf(tmp_path / "c.nc")
    ds = ar.open_dataset(tmp_path / "c.nc")
    ds = ds.assign({"rec.n": ds["rec.n"].astype("int64")})
    ds.to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert "rec" not in f.variables and f["rec.n"].dtype == np.int64
