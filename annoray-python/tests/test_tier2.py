"""equals/identical, annoray.testing, broadcast, cov/corr, pandas and dict conversions,
against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")


def build(m):
    da = m.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [20.0, 10.0], "y": [1.0, 2.0, 3.0], "lab": ("x", ["a", "b"])},
        name="a",
        attrs={"k": "v"},
    )
    ds = m.Dataset({"a": da, "b": (("y",), np.array([1.0, 2.0, 3.0]))}, attrs={"t": "g"})
    return da, ds


def test_equality_matches_xarray():
    da, ds = build(ar)
    xda, _ = build(xr)
    pairs = [
        (lambda o: o.copy(), "equals"),
        (lambda o: o.assign_attrs(z="1"), "identical"),
        (lambda o: o.rename("n"), "identical"),
        (lambda o: o.isel(x=0), "broadcast_equals"),
        (lambda o: o + 1, "equals"),
        (lambda o: o.where(o > 2), "equals"),
    ]
    for f, method in pairs:
        for name in ("equals", "identical", "broadcast_equals"):
            assert getattr(da, name)(f(da)) == getattr(xda, name)(f(xda)), (method, name)
    nan = da.where(da > 2)
    assert nan.equals(nan.copy())
    assert ds.equals(ds.copy()) and ds.identical(ds.copy())
    assert not ds.identical(ds.assign_attrs(z="1")) and ds.equals(ds.assign_attrs(z="1"))
    assert not da.equals("a")
    # Units: equal across a kind, identical only with the same unit.
    km = ar.DataArray([1.0, 2.5], dims="x", units="km")
    m = ar.DataArray([1000.0, 2500.0], dims="x", units="m")
    assert km.equals(m) and not km.identical(m)
    assert not km.equals(ar.DataArray([1.0, 2.5], dims="x", units="s"))


def test_testing_module():
    da, ds = build(ar)
    ar.testing.assert_equal(da, da.copy())
    ar.testing.assert_identical(ds, ds.copy())
    ar.testing.assert_allclose(da, da + 1e-12)
    ar.testing.assert_allclose(ds, ds.copy())
    ar.testing.assert_equal(da, da.transpose(), check_dim_order=False)
    with pytest.raises(AssertionError):
        ar.testing.assert_equal(da, da + 1)
    with pytest.raises(AssertionError):
        ar.testing.assert_identical(da, da.rename("n"))
    with pytest.raises(AssertionError):
        ar.testing.assert_allclose(da, da + 1e-3)
    with pytest.raises(AssertionError):
        ar.testing.assert_equal(da, ds)
    km = ar.DataArray([1.0, 2.5], dims="x", units="km")
    ar.testing.assert_allclose(km, ar.DataArray([1000.0, 2500.0], dims="x", units="m"))
    with pytest.raises(ar.UnitError):
        ar.testing.assert_allclose(km, ar.DataArray([1.0, 2.5], dims="x", units="s"))


def test_broadcast_and_statistics():
    da, _ = build(ar)
    xda, _ = build(xr)
    for f in (
        lambda m, a: m.broadcast(a.isel(y=0), a.isel(x=0)),
        lambda m, a: m.broadcast(a, a.isel(x=0)),
        lambda m, a: (a.isel(y=0).broadcast_like(a),),
        lambda m, a: (a.isel(y=0).broadcast_like(a, exclude="y"),),
    ):
        for r, x in zip(f(ar, da), f(xr, xda)):
            assert r.dims == x.dims
            np.testing.assert_array_equal(r.values, x.values)
    other = da * 2 + np.sin(da)
    xother = xda * 2 + np.sin(xda)
    for dim in (None, "y", ["x", "y"]):
        np.testing.assert_allclose(
            ar.cov(da, other, dim=dim).values, xr.cov(xda, xother, dim=dim).values
        )
        np.testing.assert_allclose(
            ar.cov(da, other, dim=dim, ddof=0).values, xr.cov(xda, xother, dim=dim, ddof=0).values
        )
        np.testing.assert_allclose(
            ar.corr(da, other, dim=dim).values, xr.corr(xda, xother, dim=dim).values
        )
    gappy = da.where(da != 4)
    np.testing.assert_allclose(
        ar.cov(gappy, other, dim="y").values, xr.cov(xda.where(xda != 4), xother, dim="y").values
    )
    m = ar.DataArray([1.0, 2.0, 4.0], dims="t", units="m")
    s = ar.DataArray([1.0, 3.0, 2.0], dims="t", units="s")
    assert ar.Unit(str(ar.cov(m, s).units)) == ar.Unit("m s")
    assert ar.corr(m, s).units is None


def test_pandas_conversions():
    da, ds = build(ar)
    xda, xds = build(xr)
    pd.testing.assert_frame_equal(da.to_pandas(), xda.to_pandas())
    pd.testing.assert_series_equal(da.isel(x=0).to_pandas(), xda.isel(x=0).to_pandas())
    pd.testing.assert_series_equal(da.to_series(), xda.to_series())
    pd.testing.assert_frame_equal(da.to_dataframe(), xda.to_dataframe(), check_like=True)
    order = ["y", "x"]
    pd.testing.assert_frame_equal(
        ds.to_dataframe(dim_order=order), xds.to_dataframe(dim_order=order), check_like=True
    )
    with pytest.raises(ValueError, match="unnamed"):
        da.rename(None).to_dataframe()
    s = xda.to_series().iloc[[0, 2, 5]]
    r, x = ar.DataArray.from_series(s), xr.DataArray.from_series(s)
    assert r.dims == x.dims
    np.testing.assert_array_equal(r.values, x.values)
    df = xds.to_dataframe()
    r, x = ar.Dataset.from_dataframe(df), xr.Dataset.from_dataframe(df)
    for k in x.data_vars:
        assert r[k].dims == x[k].dims
        np.testing.assert_array_equal(r[k].values, x[k].values)


def test_units_survive_pandas_and_dicts():
    m = ar.DataArray([1.0, 2.0], dims="x", coords={"x": [0.0, 1.0]}, name="d", units="m")
    s = m.to_series()
    assert s.attrs["units"] == "m"
    assert ar.DataArray.from_series(s).units == "m"
    df = m.to_dataframe()
    assert ar.Dataset.from_dataframe(df)["d"].units == "m"
    d = m.to_dict()
    assert d["attrs"]["units"] == "m"
    back = ar.DataArray.from_dict(d)
    assert back.units == "m" and "units" not in back.attrs
    ar.testing.assert_identical(back, m)


def test_dicts_match_xarray():
    da, ds = build(ar)
    xda, xds = build(xr)
    assert da.to_dict() == xda.to_dict()
    assert da.to_dict(data=False) == xda.to_dict(data=False)
    assert ds.to_dict()["data_vars"] == xds.to_dict()["data_vars"]
    ar.testing.assert_identical(ar.DataArray.from_dict(xda.to_dict()), da)
    ar.testing.assert_identical(ar.Dataset.from_dict(xds.to_dict()), ds)
