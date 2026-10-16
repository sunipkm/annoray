"""Module functions (combine_nested, combine_by_coords, cross, infer_freq, set_options,
open_dataarray, save_mfdataset, map_blocks), DataArray.to_netcdf, and live encoding."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def build(m):
    return m.Dataset(
        {"a": (("x", "y"), np.arange(6.0).reshape(2, 3))},
        coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0]},
    )


def test_combining():
    ds, xds = build(ar), build(xr)
    parts = [
        [ds.isel(x=[0], y=[0, 1]), ds.isel(x=[0], y=[2])],
        [ds.isel(x=[1], y=[0, 1]), ds.isel(x=[1], y=[2])],
    ]
    xparts = [
        [xds.isel(x=[0], y=[0, 1]), xds.isel(x=[0], y=[2])],
        [xds.isel(x=[1], y=[0, 1]), xds.isel(x=[1], y=[2])],
    ]
    r = ar.combine_nested(parts, concat_dim=["x", "y"])
    x = xr.combine_nested(xparts, concat_dim=["x", "y"])
    np.testing.assert_array_equal(r["a"].values, x["a"].values)
    shuffled = [ds.isel(x=[1]), ds.isel(x=[0])]
    r = ar.combine_by_coords(shuffled)
    np.testing.assert_array_equal(r["x"].values, [10.0, 20.0])
    merged = ar.combine_nested([ds, ds.rename_vars(a="b")], concat_dim=None)
    assert sorted(merged.data_vars) == ["a", "b"]


def test_cross_and_frequency():
    a = ar.DataArray([1.0, 0.0, 0.0], dims="c", coords={"c": ["x", "y", "z"]}, units="m")
    b = ar.DataArray([0.0, 2.0, 0.0], dims="c", coords={"c": ["x", "y", "z"]}, units="N")
    r = ar.cross(a, b, dim="c")
    x = xr.cross(xr.DataArray([1.0, 0, 0], dims="c"), xr.DataArray([0, 2.0, 0], dims="c"), dim="c")
    np.testing.assert_array_equal(r.values, x.values)
    assert ar.Unit(str(r.units)) == ar.Unit("N m")
    times = ar.DataArray(ar.date_range("2000-01-01", 4, "D"), dims="t")
    assert ar.infer_freq(times) == "D"


def test_set_options():
    with ar.set_options(keep_attrs=True, display_width=100, use_bottleneck=False):
        pass
    ar.set_options(arithmetic_join="inner")
    with pytest.raises(ValueError, match="always uses keep_attrs"):
        ar.set_options(keep_attrs=False)
    with pytest.raises(ValueError, match="unknown option"):
        ar.set_options(nonsense=1)


def test_files_and_blocks(tmp_path):
    da = ar.DataArray([1.0, 2.0], dims="x", units="m")
    da.to_netcdf(tmp_path / "unnamed.nc")
    back = ar.open_dataarray(tmp_path / "unnamed.nc")
    assert back.name is None and back.units == "m"
    named = da.rename("d")
    named.to_netcdf(tmp_path / "named.nc")
    assert ar.open_dataarray(tmp_path / "named.nc").name == "d"
    assert xr.open_dataarray(tmp_path / "unnamed.nc").name is None
    ds = build(ar)
    ar.save_mfdataset([ds.isel(x=[0]), ds.isel(x=[1])], [tmp_path / "p0.nc", tmp_path / "p1.nc"])
    assert ar.open_dataset(tmp_path / "p1.nc")["x"].values.tolist() == [20.0]
    two = ds.assign(b=ds["a"] * 2)
    two.to_netcdf(tmp_path / "two.nc")
    with pytest.raises(ValueError, match="exactly one"):
        ar.open_dataarray(tmp_path / "two.nc")
    assert ar.map_blocks(lambda o, k: o * k, ds, args=(2,))["a"].values[1, 2] == 10.0
    assert ds.map_blocks(lambda o: o + 1)["a"].values[0, 0] == 1.0


def test_encoding_is_live_and_written(tmp_path):
    src = tmp_path / "src.nc"
    xr.Dataset({"v": (("x",), np.arange(4.0))}).to_netcdf(src)
    ds = ar.open_dataset(src)
    ds["v"].encoding.update(zlib=True, complevel=6)
    assert ds["v"].encoding["zlib"] is True and ds["v"].encoding["complevel"] == 6
    ds.encoding["unlimited_dims"] = ["x"]
    out = tmp_path / "out.nc"
    ds.to_netcdf(out)
    back = xr.open_dataset(out)
    assert back["v"].encoding.get("zlib") is True or back["v"].encoding.get("compression") == "zlib"
    assert back["v"].encoding.get("complevel") == 6
    assert back.encoding.get("unlimited_dims") == {"x"}
    del ds["v"].encoding["zlib"]
    assert "zlib" not in ds["v"].encoding
    with pytest.raises(KeyError):
        del ds["v"].encoding["nonsense"]
