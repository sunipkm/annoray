"""Zarr stores: annoray reads xarray's and xarray reads annoray's, v2 and v3."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

FIXTURES = Path(__file__).parents[2] / "fixtures" / "zarr"


def same(a, x):
    """An annoray dataset against an xarray one: variables, coordinates, units, attributes."""
    assert sorted(a.data_vars) == sorted(x.data_vars)
    for k in x.variables:
        av, xv = a[k], x[k]
        assert tuple(av.dims) == tuple(xv.dims), k
        got, want = np.asarray(av.values), np.asarray(xv.values)
        if want.dtype.kind in "fc":
            np.testing.assert_allclose(got, want, equal_nan=True, err_msg=k)
        else:
            np.testing.assert_array_equal(got.astype(want.dtype), want, err_msg=k)
        if "units" in xv.attrs and xv.dtype.kind not in "Mm":
            assert av.unit == ar.Unit(xv.attrs["units"]), k
    assert a.attrs.get("title") == x.attrs.get("title")


@pytest.mark.parametrize("fmt", [2, 3])
def test_reads_xarray_stores(fmt):
    path = FIXTURES / f"xarray_v{fmt}.zarr"
    same(ar.open_zarr(path), xr.open_zarr(path))
    same(ar.open_dataset(path), xr.open_zarr(path))


@pytest.mark.parametrize("fmt", [2, 3])
def test_xarray_reads_annoray_stores(tmp_path, fmt):
    src = ar.open_dataset(FIXTURES / f"xarray_v{fmt}.nc")
    out = tmp_path / "out.zarr"
    src.to_zarr(out, zarr_format=fmt, encoding={"temp": {"chunks": (3, 2)}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # consolidated metadata: no fallback warning
        back = xr.open_zarr(out)
    same(src, back)
    assert back["temp"].encoding["chunks"] == (3, 2)
    assert back["time"].dtype.kind == "M"
    # And annoray reads its own store back.
    same(ar.open_zarr(out), back)


def test_modes_and_data_arrays(tmp_path):
    ds = ar.open_dataset(FIXTURES / "xarray_v3.nc")
    out = tmp_path / "s.zarr"
    ds.to_zarr(out)
    with pytest.raises(FileExistsError):
        ds.to_zarr(out)
    ds.to_zarr(out, mode="w", zarr_format=2)
    assert (out / ".zgroup").exists()
    with pytest.raises(NotImplementedError):
        ds.to_zarr(out, mode="a")  # pyright: ignore[reportArgumentType]
    da_out = tmp_path / "da.zarr"
    ar.to_zarr(ds["temp"], da_out)
    np.testing.assert_allclose(
        xr.open_zarr(da_out)["temp"].values, ds["temp"].values, equal_nan=True
    )
    ds.to_zarr(tmp_path / "plain.zarr", consolidated=False)
    assert not (tmp_path / "plain.zarr" / "zarr.json").read_text().count("consolidated_metadata")
