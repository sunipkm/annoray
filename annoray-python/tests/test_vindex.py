"""Vectorized (pointwise) indexing with DataArray indexers, checked against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def pair():
    v = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    x = xr.DataArray(
        v,
        dims=("z", "x", "y"),
        coords={
            "z": [5, 6],
            "x": [0.0, 10.0, 20.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "lbl": ("x", ["a", "b", "c"]),
        },
        name="v",
    )
    return ar.from_xarray(x), x


def both(values, dims, coords=None):
    x = xr.DataArray(np.asarray(values), dims=dims, coords=coords)
    return ar.from_xarray(x), x


def same(a, x):
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(x.values))
    assert tuple(a.dims) == tuple(x.dims)
    for k in x.coords:
        assert k in a.coords, k
        assert tuple(a.coords[k].dims) == tuple(x.coords[k].dims), k
        np.testing.assert_array_equal(
            np.asarray(a.coords[k].values), np.asarray(x.coords[k].values)
        )


def test_points():
    a, x = pair()
    ia, ix = both([0, 2, -1], "points")
    ja, jx = both([1, 3, 0], "points")
    same(a.isel(x=ia, y=ja), x.isel(x=ix, y=jx))


def test_outer_and_2d_indexers():
    a, x = pair()
    ia, ix = both([0, 2], "p")
    ja, jx = both([1, 2, 3], "q")
    same(a.isel(x=ia, y=ja), x.isel(x=ix, y=jx))
    ka, kx = both([[0, 1], [2, 0]], ("u", "w"))
    same(a.isel(x=ka), x.isel(x=kx))


def test_shared_dimension_and_mixed_indexers():
    a, x = pair()
    ia, ix = both([2, 1, 0, 2], "y")
    same(a.isel(x=ia), x.isel(x=ix))
    ja, jx = both([0, 1], "p")
    same(a.isel(z=1, x=ja, y=slice(1, 3)), x.isel(z=1, x=jx, y=slice(1, 3)))
    same(a.isel(z=[1, 0], y=ja), x.isel(z=[1, 0], y=jx))


def test_indexer_coordinates():
    a, x = pair()
    ia, ix = both([0, 2], "p", coords={"p": ["first", "second"]})
    same(a.isel(x=ia), x.isel(x=ix))
    # xarray indexers work directly.
    same(a.isel(x=ix), x.isel(x=ix))


def test_sel():
    a, x = pair()
    la, lx = both([20.0, 0.0], "p")
    ma, mx = both([2.9, 0.2], "p")
    same(a.sel(x=la, y=ma, method="nearest"), x.sel(x=lx, y=mx, method="nearest"))
    sa, sx = both([20.0, 10.0], "x", coords={"x": [20.0, 10.0]})
    same(a.sel(x=sa), x.sel(x=sx))
    with pytest.raises(KeyError):
        a.sel(x=ma)


def test_errors():
    a, _ = pair()
    with pytest.raises(IndexError):
        a.isel(x=ar.DataArray([5], dims="p"))
    with pytest.raises(ValueError):
        a.isel(x=ar.DataArray([0, 1], dims="p"), y=ar.DataArray([0, 1, 2], dims="p"))


def test_dataset():
    a, x = pair()
    ds, xs = ar.Dataset({"v": a, "w": a * 2.0}), xr.Dataset({"v": x, "w": x * 2.0})
    ia, ix = both([0, 2], "p")
    ja, jx = both([3, 1], "p")
    got, want = ds.isel(x=ia, y=ja), xs.isel(x=ix, y=jx)
    for k in ["v", "w"]:
        same(got[k], want[k])
