"""Variable-length (ragged) variables: read as CF contiguous ragged arrays (values along a
sample dimension, plus a count variable), written back as variable-length variables."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")
h5py = pytest.importorskip("h5py")

ROWS = [np.arange(3, dtype="i4"), np.array([5], "i4"), np.array([], "i4"), np.array([7, 8], "i4")]


def write_vlen(path):
    with nc.Dataset(path, "w") as f:
        f.createDimension("x", len(ROWS))
        t = f.createVLType(np.int32, "ragged")
        v = f.createVariable("v", t, ("x",))
        a = np.empty(len(ROWS), object)
        for i, r in enumerate(ROWS):
            a[i] = r
        v[:] = a
        v.units = "m"


def test_read_as_cf_contiguous_ragged_array(tmp_path):
    write_vlen(tmp_path / "v.nc")
    ds = ar.open_dataset(tmp_path / "v.nc")
    assert ds["v"].dims == ("v_obs",) and ds["v"].values.tolist() == [0, 1, 2, 5, 7, 8]
    assert str(ds["v"].units) == "m"
    assert ds["v_count"].dims == ("x",) and ds["v_count"].values.tolist() == [3, 1, 0, 2]
    assert ds["v_count"].attrs["sample_dimension"] == "v_obs"
    # Row i: the values after the counts of the rows before it.
    ends = np.cumsum(ds["v_count"].values)
    starts = ends - ds["v_count"].values
    assert [ds["v"].values[s:e].tolist() for s, e in zip(starts, ends, strict=True)] == [
        r.tolist() for r in ROWS
    ]


def test_h5py_ragged_floats_in_two_dimensions(tmp_path):
    with h5py.File(tmp_path / "w.h5", "w") as f:
        d = f.create_dataset("w", (2, 2), dtype=h5py.vlen_dtype(np.float64))
        d[0, 0], d[0, 1], d[1, 0], d[1, 1] = [1.5], [], [2.5, 3.5], [4.5, 5.5, 6.5]
    ds = ar.open_dataset(tmp_path / "w.h5")
    assert ds["w"].values.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    assert ds["w_count"].values.tolist() == [[1, 0], [2, 3]]


def test_written_back_as_variable_length(tmp_path):
    write_vlen(tmp_path / "v.nc")
    ds = ar.open_dataset(tmp_path / "v.nc")
    assert ds["v"].encoding["ragged_counts"] == "v_count"
    ds.to_netcdf(tmp_path / "same.nc")
    with nc.Dataset(tmp_path / "same.nc") as f:
        assert list(f.variables) == ["v"]
        assert [np.asarray(x).tolist() for x in f["v"][:]] == [r.tolist() for r in ROWS]
    # Arithmetic drops the encoding (as in xarray): the values are written in the CF form,
    # unless the pairing is given again.
    ds["v"] = ds["v"] * 2
    ds.to_netcdf(tmp_path / "cf.nc")
    with nc.Dataset(tmp_path / "cf.nc") as f:
        assert set(f.variables) == {"v", "v_count"}
    ds.to_netcdf(tmp_path / "out.nc", encoding={"v": {"ragged_counts": "v_count"}})
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert list(f.variables) == ["v"] and list(f.dimensions) == ["x"]
        got = [np.asarray(x).tolist() for x in f["v"][:]]
    assert got == [(r * 2).tolist() for r in ROWS]


def test_without_its_counts_it_is_a_plain_variable(tmp_path):
    write_vlen(tmp_path / "v.nc")
    ds = ar.open_dataset(tmp_path / "v.nc").drop_vars("v_count")
    ds.to_netcdf(tmp_path / "out.nc")
    with nc.Dataset(tmp_path / "out.nc") as f:
        assert f["v"].dimensions == ("v_obs",) and f["v"][:].tolist() == [0, 1, 2, 5, 7, 8]
