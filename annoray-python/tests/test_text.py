"""Text values are numpy's fixed-width strings (`<U{n}`), as xarray gives for numpy input,
netCDF-4 strings and Zarr fixed-width strings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.mark.parametrize(
    "data", [np.array(["a", "bcd"]), ["a", "é", ""], np.array(["日本", "a"]), np.array([["x"]])]
)
def test_in_memory_matches_xarray(data):
    a, x = ar.DataArray(data), xr.DataArray(data)
    assert a.dtype == x.dtype and a.values.dtype == x.values.dtype
    assert a.values.tolist() == x.values.tolist()
    assert a.nbytes == x.nbytes
    assert a.to_dict(data=False)["dtype"] == x.to_dict(data=False)["dtype"]


# zarr-python warns about consolidated v3 metadata when xarray writes the store.
@pytest.mark.filterwarnings("ignore:Consolidated metadata")
def test_files_match_xarray(tmp_path):
    a, x = (
        ar.open_dataset(FIXTURES / "nc" / "xr_nc4.nc"),
        xr.open_dataset(FIXTURES / "nc" / "xr_nc4.nc"),
    )
    assert a["station"].is_lazy and a["station"].dtype == x["station"].dtype
    assert a["station"].values.tolist() == x["station"].values.tolist()
    store = tmp_path / "s.zarr"
    xr.Dataset({"s": ("x", np.array(["a", "bcd"]))}).to_zarr(store)
    assert ar.open_dataset(store)["s"].values.dtype == xr.open_zarr(store)["s"].values.dtype


def test_where_xarray_gives_objects():
    # xarray keeps object arrays as given, and decodes char arrays and variable-length Zarr
    # strings to objects; annoray's text is fixed-width throughout (same values).
    a = ar.DataArray(np.array(["a", "bcd"], dtype=object))
    assert a.values.dtype == np.dtype("<U3")
    assert a.values.astype(object).tolist() == ["a", "bcd"]


def test_text_round_trips_and_converts(tmp_path):
    a = ar.DataArray(np.array(["alpha", "b"]), dims="x", name="t")
    path = tmp_path / "t.nc"
    a.to_netcdf(path)
    back = ar.open_dataarray(path)
    assert back.dtype == np.dtype("<U5") and back.values.tolist() == ["alpha", "b"]
    assert a.to_xarray().dtype == np.dtype("<U5")
    assert a.str.upper().values.tolist() == ["ALPHA", "B"]
    assert a.astype("U8").dtype.kind == "U"
