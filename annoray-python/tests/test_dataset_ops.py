"""Dataset arithmetic, comparisons, numpy ufuncs, where/isnull/map, checked against xarray."""

from __future__ import annotations

import operator

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def pair():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 4))
    a[1, 2] = np.nan
    x = xr.Dataset(
        {"a": (("y", "x"), a), "b": (("x",), rng.random(4) + 0.5)},
        coords={"x": [10, 20, 30, 40], "y": [1.0, 2.0, 3.0]},
        attrs={"title": "t"},
    )
    return ar.from_xarray(x), x


def same(a, x):
    assert sorted(a.data_vars) == sorted(x.data_vars)
    for k in x.data_vars:
        av, xv = np.asarray(a[k].values), np.asarray(x[k].values)
        assert tuple(a[k].dims) == tuple(x[k].dims), k
        if xv.dtype == bool:
            np.testing.assert_array_equal(av, xv, err_msg=k)
        else:
            np.testing.assert_allclose(av, xv, rtol=1e-12, equal_nan=True, err_msg=k)


ARITH = [operator.add, operator.sub, operator.mul, operator.truediv]


@pytest.mark.parametrize("op", ARITH)
def test_arithmetic(op):
    a, x = pair()
    same(op(a, 2.0), op(x, 2.0))
    same(op(2.0, a), op(2.0, x))
    same(op(a, a), op(x, x))
    same(op(a, a["b"]), op(x, x["b"]))
    same(a - a.mean("x"), x - x.mean("x"))
    # Attributes kept, as xarray's default keep_attrs.
    assert op(a, 2.0).attrs == op(x, 2.0).attrs == {"title": "t"}


@pytest.mark.parametrize(
    "op", [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne]
)
def test_comparisons(op):
    a, x = pair()
    same(op(a, 0.0), op(x, 0.0))
    same(op(0.0, a), op(0.0, x))
    same(op(a, a), op(x, x))


def test_logic_unary_and_ufuncs():
    a, x = pair()
    same((a > 0) & (a < 1), (x > 0) & (x < 1))
    same((a > 0) | (a < -1), (x > 0) | (x < -1))
    same(~(a > 0), ~(x > 0))
    same(-a, -x)
    same(abs(a), abs(x))
    same(a**2, x**2)
    same(np.sqrt(abs(a)), np.sqrt(abs(x)))
    same(np.exp(a), np.exp(x))
    same(np.maximum(a, 0.0), np.maximum(x, 0.0))
    same(np.greater(a, 0.0), np.greater(x, 0.0))


def test_alignment_and_common_variables():
    a, x = pair()
    a2, x2 = a.isel(x=slice(1, 3))[["a"]], x.isel(x=slice(1, 3))[["a"]]
    same(a + a2, x + x2)
    np.testing.assert_array_equal((a + a2)["x"].values, (x + x2)["x"].values)


def test_where_isnull_map():
    a, x = pair()
    same(a.where(a > 0), x.where(x > 0))
    same(a.where(a["b"] > 0.7), x.where(x["b"] > 0.7))
    same(a.where(a > 0, 0.0), x.where(x > 0, 0.0))
    same(a.isnull(), x.isnull())
    same(a.notnull(), x.notnull())
    same(a.map(lambda v: v * 3), x.map(lambda v: v * 3))
    assert a.map(lambda v: v * 3, keep_attrs=True).attrs == {"title": "t"}
    with pytest.raises(TypeError):
        a.map(lambda v: 1)


def test_units_and_refusals():
    ds = ar.Dataset(
        {
            "h": ar.DataArray([1.0, 2.0], dims="x", attrs={"units": "m"}),
            "t": ar.DataArray([3.0, 4.0], dims="x", attrs={"units": "s"}),
        }
    )
    speed = ds["h"] / ds["t"]
    assert speed.unit is not None and speed.unit.is_convertible(ar.Unit("m/s"))
    assert (ds * 2)["h"].unit == ar.Unit("m")
    with pytest.raises(ar.UnitError):
        _ = ds + 1  # a bare number added to metres
    np.testing.assert_array_equal((ds > 0)["h"].values, [True, True])
    text = ar.Dataset({"s": ar.DataArray(np.array(["a", "b"]), dims="x")})
    with pytest.raises(TypeError):
        _ = text * 2  # text cannot be multiplied (as in xarray)
