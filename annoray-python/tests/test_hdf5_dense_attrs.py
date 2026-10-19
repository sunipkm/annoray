"""Attributes in dense storage (a fractal heap indexed by a v2 B-tree): read, including the
huge heap objects HDF5 uses for large attributes, and written for attributes over 64 KiB."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import annoray as ar

h5py = pytest.importorskip("h5py")
nc = pytest.importorskip("netCDF4")


def test_read_huge_attributes_from_h5py(tmp_path):
    with h5py.File(tmp_path / "h.h5", "w", libver="latest") as f:
        d = f.create_dataset("v", data=np.arange(3.0))
        d.attrs["big"] = np.arange(20000.0)
        for i in range(10):
            d.attrs[f"a{i}"] = i
        f.attrs["gbig"] = np.arange(30000, dtype="i4")
    ds = ar.open_dataset(tmp_path / "h.h5")
    np.testing.assert_array_equal(ds["v"].attrs["big"], np.arange(20000.0))
    assert ds["v"].attrs["a9"] == 9
    np.testing.assert_array_equal(ds.attrs["gbig"], np.arange(30000))


def test_read_huge_attributes_from_netcdf4(tmp_path):
    with nc.Dataset(tmp_path / "n.nc", "w") as f:
        f.createDimension("x", 3)
        v = f.createVariable("v", "f8", ("x",))
        v.big = np.arange(20000.0)
        v.s = "x" * 100000
    ds = ar.open_dataset(tmp_path / "n.nc")
    assert len(ds["v"].attrs["big"]) == 20000 and len(ds["v"].attrs["s"]) == 100000


def big_dataset():
    attrs: dict[str, Any] = {f"a{i}": i for i in range(40)}
    attrs.update({"big": np.arange(20000.0), "units": "m", "s": "y" * 70000})
    return ar.Dataset(
        {"v": ("x", np.arange(3.0), attrs)},
        attrs={"gbig": np.arange(30000, dtype="i4"), "title": "t"},
    )


def test_write_attributes_over_64_kib(tmp_path):
    big_dataset().to_netcdf(tmp_path / "d.nc")
    with h5py.File(tmp_path / "d.nc") as f:
        np.testing.assert_array_equal(f["v"].attrs["big"], np.arange(20000.0))
        assert f["v"].attrs["s"] == b"y" * 70000  # netCDF text: fixed-length bytes
        assert len(f.attrs["gbig"]) == 30000
        # The dimension references survive dense storage.
        assert len(f["v"].dims[0]) == 1
    with nc.Dataset(tmp_path / "d.nc") as f:
        assert f["v"].ncattrs()[:2] == ["a0", "a1"]
        assert len(f["v"].big) == 20000 and f["v"].units == "m"
        assert f.ncattrs() == ["gbig", "title"]
    back = ar.open_dataset(tmp_path / "d.nc")
    assert list(back["v"].attrs)[:3] == ["a0", "a1", "a2"]
    assert str(back["v"].units) == "m"
    np.testing.assert_array_equal(back.attrs["gbig"], np.arange(30000))


def test_hdf5_can_keep_modifying_dense_attributes(tmp_path):
    big_dataset().to_netcdf(tmp_path / "d.nc")
    with h5py.File(tmp_path / "d.nc", "a") as f:
        a = f["v"].attrs
        a["added"] = 5
        del a["a3"]
        del a["big"]
        a["big2"] = np.ones(30000)
        for i in range(300):
            a[f"m{i}"] = np.arange(i)
    with nc.Dataset(tmp_path / "d.nc", "a") as f:
        f["v"].more = "x" * 80000
    v = ar.open_dataset(tmp_path / "d.nc")["v"]
    assert v.attrs["added"] == 5 and "a3" not in v.attrs and "big" not in v.attrs
    assert v.attrs["m299"][-1] == 298 and len(v.attrs["more"]) == 80000
