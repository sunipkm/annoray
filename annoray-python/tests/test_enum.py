"""Enumeration types (netCDF-4 `NC_ENUM`, HDF5 enums): values read as the base integers, the
names kept in `encoding["dtype"].metadata` (as xarray does) and written back as the type;
h5py's booleans (an enum of FALSE and TRUE) read as booleans."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")
h5py = pytest.importorskip("h5py")

CLOUDS = {"clear": 0, "cumulus": 1, "missing": 255}


def write_enum(path):
    with nc.Dataset(path, "w") as f:
        t = f.createEnumType(np.uint8, "cloud_t", CLOUDS)
        f.createDimension("x", 3)
        v = f.createVariable("cloud", t, ("x",), fill_value=255)
        v[:] = np.array([0, 1, 255], "u1")


@pytest.mark.parametrize("lazy", [True, False])
def test_read_enum(tmp_path, lazy):
    write_enum(tmp_path / "e.nc")
    v = ar.open_dataset(tmp_path / "e.nc", lazy=lazy)["cloud"]
    np.testing.assert_array_equal(v.values, [0, 1, np.nan])
    dtype = v.encoding["dtype"]
    assert dtype == np.uint8
    assert dtype.metadata == {"enum": CLOUDS, "enum_name": "cloud_t"}
    raw = ar.open_dataset(tmp_path / "e.nc", mask_and_scale=False, lazy=lazy)["cloud"]
    assert raw.dtype == np.uint8 and raw.values.tolist() == [0, 1, 255]


def test_enum_round_trip(tmp_path):
    write_enum(tmp_path / "e.nc")
    ar.open_dataset(tmp_path / "e.nc").to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        t = f["cloud"].datatype
        assert isinstance(t, nc.EnumType)
        assert t.name == "cloud_t" and t.enum_dict == CLOUDS
        assert f["cloud"][:].mask.tolist() == [False, False, True]
    again = ar.open_dataset(tmp_path / "out.nc")["cloud"].encoding["dtype"]
    assert again.metadata == {"enum": CLOUDS, "enum_name": "cloud_t"}


def test_new_enum_from_encoding(tmp_path):
    c = ar.DataArray(np.array([0, 2, 1], "i2"), dims="x", name="c")
    members = {"a": 0, "b": 1, "c": 2}
    c.encoding = {"dtype": np.dtype("i2", metadata={"enum": members, "enum_name": "abc_t"})}
    ar.Dataset({"c": c}).to_netcdf(tmp_path / "n.nc")
    with nc.Dataset(tmp_path / "n.nc") as f:
        assert f.enumtypes["abc_t"].enum_dict == members
        assert f["c"][:].tolist() == [0, 2, 1]


def test_unnamed_enum_gets_a_name(tmp_path):
    c = ar.DataArray(np.array([0, 1], "u1"), dims="x", name="c")
    c.encoding = {"dtype": np.dtype("u1", metadata={"enum": {"no": 0, "yes": 1}})}
    ar.Dataset({"c": c}).to_netcdf(tmp_path / "n.nc")
    with nc.Dataset(tmp_path / "n.nc") as f:
        assert list(f.enumtypes) == ["enum0_t"]


def test_enum_member_must_fit(tmp_path):
    c = ar.DataArray(np.array([0, 1], "i1"), dims="x", name="c")
    c.encoding = {"dtype": np.dtype("i1", metadata={"enum": {"a": 0, "big": 300}})}
    with pytest.raises(Exception, match="300"):
        ar.Dataset({"c": c}).to_netcdf(tmp_path / "n.nc")


def test_enum_needs_integer_dtype():
    c = ar.DataArray(np.array([0.0, 1.0]), dims="x")
    with pytest.raises(ValueError, match="integer"):
        c.encoding = {"dtype": np.dtype("f8", metadata={"enum": {"a": 0}})}
        ar.Dataset({"c": c}).to_netcdf("unused.nc")


@pytest.mark.parametrize("lazy", [True, False])
def test_h5py_bool_reads_as_bool(tmp_path, lazy):
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f["b"] = np.array([[True, False], [False, True]])
    v = ar.open_dataset(tmp_path / "b.h5", lazy=lazy)["b"]
    assert v.dtype == np.bool_
    assert v.values.tolist() == [[True, False], [False, True]]
    assert getattr(v.encoding.get("dtype"), "metadata", None) is None
