"""clip, round, isin and dot, checked against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

NAN = np.nan
RNG = np.random.default_rng(11)


def pair(values=None, units=None):
    v = RNG.standard_normal((3, 4)) * 10 if values is None else values
    x = xr.DataArray(
        v,
        dims=("y", "x"),
        coords={"y": [1.0, 2.0, 3.0], "x": [10, 20, 30, 40]},
        name="v",
        attrs={} if units is None else {"units": units},
    )
    return ar.from_xarray(x), x


def same(a, x):
    av, xv = np.asarray(a.values), np.asarray(x.values)
    assert av.dtype == xv.dtype, (av.dtype, xv.dtype)
    np.testing.assert_allclose(av.astype(float), xv.astype(float), rtol=1e-12, equal_nan=True)
    assert tuple(a.dims) == tuple(x.dims)


@pytest.mark.parametrize("bounds", [(-5, 5), (None, 0), (2.5, None), (5, -5), (NAN, None)])
def test_clip_scalars(bounds):
    v = RNG.standard_normal((3, 4)) * 10
    v[0, 1] = NAN
    a, x = pair(v)
    same(a.clip(*bounds), x.clip(*bounds))


def test_clip_types_units_arrays():
    ints = np.arange(12, dtype="int32").reshape(3, 4)
    a, x = pair(ints)
    same(a.clip(2, 9), x.clip(2, 9))
    same(a.clip(2.5, 9), x.clip(2.5, 9))
    f, xf = pair(RNG.standard_normal((3, 4)).astype("float32"))
    same(f.clip(-0.5, 0.5), xf.clip(-0.5, 0.5))
    lo = ar.DataArray([0.0, -1.0, 2.0], dims="y")
    a, x = pair()
    same(a.clip(min=lo), x.clip(min=xr.DataArray([0.0, -1.0, 2.0], dims="y")))
    t, _ = pair(np.full((3, 4), 30.0), units="degC")
    got = t.clip(max=ar.Quantity(293.15, "K"))
    np.testing.assert_allclose(got.values, 20.0)
    assert got.unit == ar.Unit("degC")
    assert t.clip(0.0, keep_attrs=False).attrs == {}
    with pytest.raises(ar.UnitError):
        t.clip(ar.Quantity(1.0, "m"))


@pytest.mark.parametrize("decimals", [0, 1, 2, -1])
def test_round(decimals):
    v = np.array([[0.5, 1.5, 2.5, -0.5], [15.0, 25.0, 1.2345, NAN], [-2.675, 3.14159, 1e6, 0.05]])
    a, x = pair(v)
    same(a.round(decimals), x.round(decimals))
    ints = np.array([[15, 25, -15, 14], [5, 150, 249, 251], [0, 1, 2, 3]], dtype="int64")
    a, x = pair(ints)
    same(a.round(decimals), x.round(decimals))
    assert round(a, decimals).dims == a.dims


def test_isin():
    v = np.array([[1.0, 2.0, NAN, 3.0], [4.0, 5.0, 6.0, 7.0], [2.0, 2.0, 0.0, -1.0]])
    a, x = pair(v)
    for elems in [[2.0, 7.0], np.array([2, 3]), [0.0, NAN], 5.0]:
        same(a.isin(elems), x.isin(elems))
    # A set is its elements (numpy's isin takes a set as one object and matches nothing).
    np.testing.assert_array_equal(a.isin({0.0, NAN}).values, a.isin([0.0, NAN]).values)
    m, _ = pair(v, units="m")
    np.testing.assert_array_equal(m.isin([ar.Quantity(0.002, "km")]).values, v == 2.0)
    s = ar.DataArray(np.array(["a", "b", "c"]), dims="x")
    xs = xr.DataArray(np.array(["a", "b", "c"]), dims="x")
    same(s.isin(["b", "c"]), xs.isin(["b", "c"]))
    t = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    same(ar.DataArray(t, dims="x").isin([t[1]]), xr.DataArray(t, dims="x").isin([t[1]]))


def test_dot():
    a, x = pair()
    b = RNG.standard_normal((4, 2))
    bx = xr.DataArray(b, dims=("x", "z"), coords={"x": [10, 20, 30, 40]})
    ba = ar.from_xarray(bx)
    same(a.dot(ba), x.dot(bx))
    same(a @ ba, x @ bx)
    same(ar.dot(a, ba, a), xr.dot(x, bx, x))
    same(a.dot(a, dim=...), x.dot(x, dim=...))
    same(ar.dot(a, a, dim="y"), xr.dot(x, x, dim="y"))
    am, _ = pair(units="m")
    out = ar.dot(am, ar.DataArray(bx.values, dims=("x", "z"), attrs={"units": "s"}))
    assert out.unit is not None and out.unit.is_convertible(ar.Unit("m s"))
    v = x.values.copy()
    v[0, 0] = NAN
    a2, x2 = pair(v)
    same(a2.dot(ba), x2.dot(bx))


def test_datasets():
    a, x = pair()
    ds, xs = ar.Dataset({"v": a}), xr.Dataset({"v": x})
    same(ds.round(1)["v"], xs.round(1)["v"])
    same(ds.clip(-1, 1)["v"], xs.clip(-1, 1)["v"])
    same(ds.isin([0.0])["v"], xs.isin([0.0])["v"])
