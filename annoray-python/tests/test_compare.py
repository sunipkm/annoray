"""Comparison and logical operators, checked against xarray (units checked on their own)."""

from __future__ import annotations

import operator

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def pair():
    v = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 2.0, 0.0, -1.0]])
    x = xr.DataArray(v, dims=("y", "x"), coords={"x": [10, 20, 30, 40]}, name="v")
    return ar.from_xarray(x), x


def same(a, x):
    assert np.asarray(a.values).dtype == bool
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(x.values))
    assert tuple(a.dims) == tuple(x.dims)


OPS = [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne]


@pytest.mark.parametrize("op", OPS)
def test_operators_match_xarray(op):
    a, x = pair()
    same(op(a, 2.0), op(x, 2.0))
    same(op(2.0, a), op(2.0, x))  # reflected
    same(op(a, a), op(x, x))  # NaN != NaN
    same(op(np.float64(2.0), a), op(np.float64(2.0), x))  # numpy scalar on the left


@pytest.mark.parametrize(
    "name", ["less", "less_equal", "greater", "greater_equal", "equal", "not_equal"]
)
def test_numpy_ufuncs(name):
    a, x = pair()
    f = getattr(np, name)
    same(f(a, 2.0), f(x, 2.0))
    same(f(0.0, a), f(0.0, x))


def test_masks_and_where():
    a, x = pair()
    m, mx = (a > 1) & (a < 5), (x > 1) & (x < 5)
    same(m, mx)
    same((a > 4) | (a < 1), (x > 4) | (x < 1))
    same((a > 1) ^ (a < 3), (x > 1) ^ (x < 3))
    same(~(a > 1), ~(x > 1))
    same(np.logical_and(a > 1, a < 5), np.logical_and(x > 1, x < 5))
    same(np.logical_not(a > 1), np.logical_not(x > 1))
    got, want = a.where(m), x.where(mx)
    np.testing.assert_array_equal(got.values, want.values)
    with pytest.raises(TypeError):
        _ = ~a  # not booleans (numpy raises TypeError too)


def test_text_dates_alignment():
    s, sx = (
        ar.DataArray(np.array(["a", "b", "c"]), dims="x"),
        xr.DataArray(np.array(["a", "b", "c"]), dims="x"),
    )
    same(s == "b", sx == "b")
    same(s < "b", sx < "b")
    t = np.array(["2000-01-01", "NaT", "2000-01-03"], dtype="datetime64[ns]")
    ta, tx = ar.DataArray(t, dims="t"), xr.DataArray(t, dims="t")
    pivot = np.datetime64("2000-01-02", "ns")
    same(ta > pivot, tx > pivot)
    same(ta != pivot, tx != pivot)
    a = ar.DataArray([1.0, 2.0, 3.0], dims="x", coords={"x": [1, 2, 3]})
    b = ar.DataArray([2.0, 2.0], dims="x", coords={"x": [2, 3]})
    xa = xr.DataArray([1.0, 2.0, 3.0], dims="x", coords={"x": [1, 2, 3]})
    xb = xr.DataArray([2.0, 2.0], dims="x", coords={"x": [2, 3]})
    same(a >= b, xa >= xb)


def test_units():
    length = ar.DataArray([500.0, 1500.0], dims="x", attrs={"units": "m"})
    np.testing.assert_array_equal((length > ar.Quantity(1.0, "km")).values, [False, True])
    assert (length > ar.Quantity(1.0, "km")).unit is None
    flux = ar.DataArray([-1.0, 2.0], dims="x", attrs={"units": "W m-2"})
    np.testing.assert_array_equal((flux > 0).values, [False, True])
    np.testing.assert_array_equal((0 < flux).values, [False, True])
    with pytest.raises(ar.UnitError):
        _ = flux > 5
    t = ar.DataArray([-5.0, 5.0], dims="x", attrs={"units": "degC"})
    with pytest.raises(ar.UnitError):
        _ = t > 0  # 0 °C is not 0 °F: a bare zero is ambiguous here
    np.testing.assert_array_equal((t > ar.Quantity(32.0, "degF")).values, [False, True])
    with pytest.raises(ar.UnitError):
        _ = length > ar.Quantity(1.0, "s")
