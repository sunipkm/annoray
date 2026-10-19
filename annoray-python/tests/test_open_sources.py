"""Opening files held in memory (bytes-like objects) and file-like objects, and
`decode_coords="all"`."""

from __future__ import annotations

import io

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")


@pytest.fixture(params=["NETCDF4", "NETCDF3_64BIT_OFFSET"])
def nc_bytes(tmp_path, request):
    path = tmp_path / "s.nc"
    with nc.Dataset(path, "w", format=request.param) as f:
        f.createDimension("x", 3)
        v = f.createVariable("v", "f4", ("x",))
        v[:] = [1, 2, 3]
        v.units = "m"
        f.createVariable("x", "i4", ("x",))[:] = [10, 20, 30]
    return path.read_bytes()


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview, io.BytesIO])
def test_open_from_memory(nc_bytes, wrap, lazy):
    ds = ar.open_dataset(wrap(nc_bytes), lazy=lazy)
    assert ds["v"].values.tolist() == [1, 2, 3] and str(ds["v"].units) == "m"
    assert ds["x"].values.tolist() == [10, 20, 30]


def test_file_object_is_read_from_the_start(tmp_path, nc_bytes):
    path = tmp_path / "f.nc"
    path.write_bytes(nc_bytes)
    with open(path, "rb") as f:
        f.read(10)
        ds = ar.open_dataset(f)
    assert ds["v"].values.tolist() == [1, 2, 3]


def test_text_mode_file_is_an_error(tmp_path, nc_bytes):
    path = tmp_path / "f.nc"
    path.write_bytes(nc_bytes)
    with open(path, encoding="latin-1") as f, pytest.raises(ValueError, match="binary"):
        ar.open_dataset(f)  # pyright: ignore[reportArgumentType]


def test_open_datatree_from_bytes(nc_bytes):
    tree = ar.open_datatree(nc_bytes)
    assert tree.to_dataset()["v"].values.tolist() == [1, 2, 3]


def test_bytes_of_another_format_is_an_error():
    with pytest.raises(Exception, match="not a netCDF or HDF5"):
        ar.open_dataset(b"\x00" * 64)


def write_cf(path, *, missing_crs=False):
    with nc.Dataset(path, "w") as f:
        f.createDimension("x", 3)
        f.createDimension("y", 2)
        f.createDimension("nv", 2)
        v = f.createVariable("v", "f8", ("y", "x"))
        v[:] = np.zeros((2, 3))
        v.grid_mapping = "crs"
        v.cell_measures = "area: cellarea"
        if not missing_crs:
            crs = f.createVariable("crs", "i4", ())
            crs.grid_mapping_name = "latitude_longitude"
        f.createVariable("cellarea", "f8", ("y", "x"))[:] = np.ones((2, 3))
        x = f.createVariable("x", "f8", ("x",))
        x[:] = [1, 2, 3]
        x.bounds = "x_bnds"
        f.createVariable("x_bnds", "f8", ("x", "nv"))[:] = np.zeros((3, 2))


def test_decode_coords_all(tmp_path):
    write_cf(tmp_path / "cf.nc")
    ds = ar.open_dataset(tmp_path / "cf.nc", decode_coords="all")
    assert sorted(ds.coords) == ["cellarea", "crs", "x", "x_bnds"]
    assert ds["v"].encoding["grid_mapping"] == "crs"
    assert ds["v"].encoding["cell_measures"] == "area: cellarea"
    assert ds["x"].encoding["bounds"] == "x_bnds"
    assert "grid_mapping" not in ds["v"].attrs and "bounds" not in ds["x"].attrs
    # The default (and "coordinates") leave them as attributes and data variables.
    for opt in (True, "coordinates"):
        plain = ar.open_dataset(tmp_path / "cf.nc", decode_coords=opt)
        assert sorted(plain.coords) == ["x"] and plain["v"].attrs["grid_mapping"] == "crs"


def test_decode_coords_all_round_trip(tmp_path):
    write_cf(tmp_path / "cf.nc")
    ar.open_dataset(tmp_path / "cf.nc", decode_coords="all").to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert f["v"].grid_mapping == "crs" and f["v"].cell_measures == "area: cellarea"
        assert f["x"].bounds == "x_bnds"
        # Referenced variables are not listed as coordinates.
        assert "coordinates" not in f["v"].ncattrs() and "coordinates" not in f.ncattrs()


def test_decode_coords_all_keeps_attribute_naming_a_missing_variable(tmp_path):
    write_cf(tmp_path / "cf.nc", missing_crs=True)
    with pytest.warns(UserWarning, match="crs"):
        ds = ar.open_dataset(tmp_path / "cf.nc", decode_coords="all")
    # xarray drops the attribute; it stays an attribute here.
    assert ds["v"].attrs["grid_mapping"] == "crs" and "grid_mapping" not in ds["v"].encoding
    assert "cellarea" in ds.coords


def test_decode_coords_all_from_decode_cf(tmp_path):
    write_cf(tmp_path / "cf.nc")
    raw = ar.open_dataset(tmp_path / "cf.nc", decode_cf=False)
    ds = ar.decode_cf(raw, decode_coords="all")
    assert "crs" in ds.coords


def test_bad_decode_coords():
    with pytest.raises(ValueError, match="decode_coords"):
        ar.open_dataset(b"", decode_coords="some")  # pyright: ignore[reportArgumentType]
