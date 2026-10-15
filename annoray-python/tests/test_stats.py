"""quantile, idxmin/idxmax, weighted and coarsen, checked against xarray on the same data."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

NAN = np.nan
RNG = np.random.default_rng(3)


def pair(values=None):
    v = RNG.standard_normal((4, 6)) if values is None else values
    x = xr.DataArray(
        v,
        dims=("y", "x"),
        coords={
            "y": [10.0, 20.0, 30.0, 40.0],
            "x": np.arange(6.0) * 2,
            "t": ("x", np.arange(6).astype("datetime64[D]").astype("datetime64[ns]")),
        },
        name="v",
        attrs={"units": "m"},
    )
    return ar.from_xarray(x), x


def same(a, x, coords=True):
    np.testing.assert_allclose(
        np.asarray(a.values, float), np.asarray(x.values, float), rtol=0, atol=0, equal_nan=True
    )
    assert tuple(a.dims) == tuple(x.dims)
    if coords:
        for k in a.coords:
            av, xv = np.asarray(a.coords[k].values), np.asarray(x.coords[k].values)
            if av.dtype.kind in "fc":
                np.testing.assert_allclose(av, xv, equal_nan=True)
            else:
                np.testing.assert_array_equal(av, xv)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("q", [0.5, [0.1, 0.25, 0.9], 0.0, 1.0])
@pytest.mark.parametrize("dim", ["x", None, ["y", "x"]])
def test_quantile(method, q, dim):
    values = RNG.standard_normal((4, 6))
    values[1, 2] = NAN
    a, x = pair(values)
    same(a.quantile(q, dim=dim, method=method), x.quantile(q, dim=dim, method=method))


def test_quantile_skipna_and_units():
    values = RNG.standard_normal((4, 6))
    values[1, 2] = NAN
    a, x = pair(values)
    same(a.quantile(0.3, "x", skipna=False), x.quantile(0.3, "x", skipna=False))
    assert a.quantile(0.5, "x").units == "m"
    with pytest.raises(ValueError):
        a.quantile(1.5)


@pytest.mark.parametrize("which", ["idxmin", "idxmax"])
@pytest.mark.parametrize("skipna", [True, False])
def test_idx(which, skipna):
    values = RNG.standard_normal((4, 6))
    values[1, 2] = NAN
    values[3, :] = NAN
    a, x = pair(values)
    same(getattr(a, which)("x", skipna=skipna), getattr(x, which)("x", skipna=skipna))
    same(getattr(a, which)("y", skipna=skipna), getattr(x, which)("y", skipna=skipna))


@pytest.mark.parametrize("op", ["sum_of_weights", "sum", "mean", "var", "std"])
@pytest.mark.parametrize("dim", ["y", None])
def test_weighted(op, dim):
    values = RNG.standard_normal((4, 6))
    values[0, 1] = NAN
    a, x = pair(values)
    w = xr.DataArray([1.0, 2.0, 0.5, 0.0], dims="y", coords={"y": x.y})
    got = getattr(a.weighted(ar.from_xarray(w)), op)(dim)
    want = getattr(x.weighted(w), op)(dim)
    np.testing.assert_allclose(
        np.asarray(got.values), np.asarray(want.values), rtol=1e-12, equal_nan=True
    )
    assert tuple(got.dims) == tuple(want.dims)


def test_weighted_units_and_nan_weights():
    a, _ = pair()
    w = ar.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims="y", units="kg")
    assert a.weighted(w).mean("y").units == "m"
    assert a.weighted(w).sum("y").unit.is_convertible("kg m")
    assert a.weighted(w).var("y").unit.is_convertible("m2")
    with pytest.raises(ValueError, match="missing"):
        a.weighted(ar.DataArray(np.array([1.0, NAN, 1.0, 1.0]), dims="y"))


@pytest.mark.parametrize(
    "kw",
    [
        {"x": 2},
        {"x": 3, "y": 2},
        {"x": 4, "boundary": "trim"},
        {"x": 4, "boundary": "trim", "side": "right"},
        {"x": 4, "boundary": "pad"},
        {"x": 4, "boundary": "pad", "side": "right"},
    ],
)
@pytest.mark.parametrize("op", ["mean", "sum", "max", "min", "std", "median", "count"])
def test_coarsen(kw, op):
    values = RNG.standard_normal((4, 6))
    values[2, 3] = NAN
    a, x = pair(values)
    got = getattr(a.coarsen(**kw), op)()
    want = getattr(x.coarsen(**kw), op)()
    np.testing.assert_allclose(
        np.asarray(got.values, float), np.asarray(want.values, float), rtol=1e-12, equal_nan=True
    )
    assert tuple(got.dims) == tuple(want.dims)
    for k in ("x", "y"):
        np.testing.assert_allclose(got.coords[k].values, want.coords[k].values, equal_nan=True)
    np.testing.assert_array_equal(got.coords["t"].values, want.coords["t"].values)


def test_coarsen_exact_and_datasets():
    a, x = pair()
    with pytest.raises(ValueError, match="multiple"):
        a.coarsen(x=4)
    ds, xd = a.to_dataset(), x.to_dataset()
    same(ds.coarsen(x=2).mean()["v"], xd.coarsen(x=2).mean()["v"])
    same(ds.quantile(0.5, "x")["v"], xd.quantile(0.5, "x")["v"])
    same(ds.idxmax("x")["v"], xd.idxmax("x")["v"])
