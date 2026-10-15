"""squeeze, expand_dims, swap_dims, drop_sel, sortby, assign_coords, diff, shift, roll,
cumsum, cumprod, checked against xarray on the same data."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

NAN = np.nan


def pair():
    x = xr.DataArray(
        [[1.0, NAN, 2.0, 3.0], [4.0, 5.0, NAN, 7.0]],
        dims=("y", "x"),
        coords={"x": [3.0, 1.0, NAN, 2.0], "y": [10.0, 20.0], "lbl": ("x", list("abcd"))},
        name="v",
        attrs={"units": "m"},
    )
    return ar.from_xarray(x), x


def same(a, x):
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(x.values))
    assert tuple(a.dims) == tuple(x.dims)
    for k in x.coords:
        if k in a.coords:
            np.testing.assert_array_equal(
                np.asarray(a.coords[k].values), np.asarray(x.coords[k].values)
            )


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d.cumsum("x"),
        lambda d: d.diff("x"),
        lambda d: d.shift(x=1),
        lambda d: d.shift(x=-2),
        lambda d: d.roll(x=1),
        lambda d: d.roll(x=-3, roll_coords=True),
        lambda d: d.sortby("x"),
        lambda d: d.sortby("y", ascending=False),
        lambda d: d.isel(y=slice(0, 1)).squeeze(),
        lambda d: d.isel(y=0).expand_dims("y"),
        lambda d: d.swap_dims(x="lbl"),
        lambda d: d.drop_sel(y=[20.0]),
        lambda d: d.assign_coords(y=[1.0, 2.0]),
    ],
)
def test_matches_xarray(op):
    a, x = pair()
    same(op(a), op(x))


def test_diff_lower_labels_every_step():
    # xarray applies `label` to the first difference only (its recursion drops it); annoray
    # labels every step, so n=2 "lower" keeps the first len - 2 coordinates.
    a, x = pair()
    d = a.diff("x", n=2, label="lower")
    np.testing.assert_array_equal(d.values, x.diff("x", n=2, label="lower").values)
    np.testing.assert_array_equal(d.coords["x"].values, [3.0, 1.0])


def test_units_follow():
    a, _ = pair()
    assert a.cumsum("x").units == "m"
    with pytest.raises(ar.UnitError):
        a.cumprod("x")
    c = ar.DataArray(np.array([10.0, 12.5]), dims="t", units="degC")
    step = c.diff("t")
    assert step.unit is not None and step.unit.is_convertible("K")
    np.testing.assert_allclose(step.values, [2.5])


def test_dataset_versions():
    a, x = pair()
    ds, xd = a.to_dataset(), x.to_dataset()
    for op in (
        lambda d: d.cumsum("x"),
        lambda d: d.diff("x"),
        lambda d: d.shift(x=1),
        lambda d: d.roll(x=2),
        lambda d: d.sortby("x"),
        lambda d: d.drop_sel(y=[10.0]),
        lambda d: d.swap_dims(x="lbl"),
    ):
        same(op(ds)["v"], op(xd)["v"])


def test_errors():
    a, _ = pair()
    with pytest.raises(ValueError):
        a.squeeze("x")
    with pytest.raises(ValueError, match="label"):
        a.diff("x", label="middle")  # pyright: ignore[reportArgumentType]
    with pytest.raises(KeyError):
        a.drop_sel(y=[99.0])
