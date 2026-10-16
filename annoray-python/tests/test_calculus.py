"""differentiate, integrate, cumulative_integrate, polyfit and polyval, checked against xarray."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

RNG = np.random.default_rng(7)


def pair(values=None, units=None):
    v = RNG.standard_normal((3, 7)) if values is None else values
    x = xr.DataArray(
        v,
        dims=("y", "x"),
        coords={
            "y": [10.0, 20.0, 30.0],
            "x": [0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 7.0],
            "t": ("x", np.array([0, 1, 3, 4, 8, 9, 14], "datetime64[D]").astype("datetime64[ns]")),
        },
        name="v",
        attrs={} if units is None else {"units": units},
    )
    return ar.from_xarray(x), x


def close(a, x, rtol=1e-12):
    np.testing.assert_allclose(
        np.asarray(a.values, float),
        np.asarray(x.values, float),
        rtol=rtol,
        atol=1e-12,
        equal_nan=True,
    )
    assert tuple(a.dims) == tuple(x.dims)


@pytest.mark.parametrize("edge_order", [1, 2])
def test_differentiate(edge_order):
    a, x = pair()
    close(a.differentiate("x", edge_order=edge_order), x.differentiate("x", edge_order=edge_order))
    v = a.values.copy()
    v[1, 3] = np.nan
    a, x = pair(v)
    close(a.differentiate("x", edge_order), x.differentiate("x", edge_order))


def test_differentiate_along_times():
    a, x = pair()
    a2 = a.swap_dims({"x": "t"})
    x2 = x.swap_dims({"x": "t"})
    close(a2.differentiate("t"), x2.differentiate("t", datetime_unit="s"))
    close(a2.differentiate("t", datetime_unit="D"), x2.differentiate("t", datetime_unit="D"))
    assert a2.differentiate("t", datetime_unit="h").unit == ar.Unit("1/h")
    with pytest.raises(ValueError):
        a2.differentiate("t", datetime_unit="M")
    # datetime_unit takes times only: "m" is numpy's minute, as is "min".
    for m in ("m", "min"):
        close(a2.differentiate("t", datetime_unit=m), x2.differentiate("t", datetime_unit="m"))
    # A Unit object works like a string.
    close(
        a2.differentiate("t", datetime_unit=ar.Unit("h")), x2.differentiate("t", datetime_unit="h")
    )
    assert a2.integrate("t", datetime_unit=ar.Unit("h")).unit == ar.Unit("h")
    with pytest.raises(ar.UnitError):
        a2.integrate("t", datetime_unit=ar.Unit("kg"))


def test_integrals():
    a, x = pair()
    close(a.integrate("x"), x.integrate("x"))
    close(a.integrate(["y", "x"]), x.integrate(["y", "x"]))
    close(a.cumulative_integrate("x"), x.cumulative_integrate("x"))
    close(a.cumulative_integrate(["x", "y"]), x.cumulative_integrate(["x", "y"]))
    a2, x2 = a.swap_dims({"x": "t"}), x.swap_dims({"x": "t"})
    close(a2.integrate("t", datetime_unit="h"), x2.integrate("t", datetime_unit="h"))


def test_units():
    a, _ = pair(units="m")
    a = a.assign_coords(x=ar.DataArray(a.coords["x"].values, dims="x", attrs={"units": "s"}))
    assert a.differentiate("x").unit.is_convertible(ar.Unit("m/s"))
    assert a.integrate("x").unit.is_convertible(ar.Unit("m s"))
    c, _ = pair(units="degC")
    with pytest.raises(ar.UnitError):
        c.integrate("x")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"full": True},
        {"cov": True},
        {"cov": "unscaled"},
        {"w": "y"},
        {"skipna": False},
        {"rcond": 1e-3},
    ],
)
def test_polyfit(kwargs):
    v = RNG.standard_normal((7, 3))
    v[2, 1] = np.nan
    x = xr.DataArray(v, dims=("x", "y"), coords={"x": np.linspace(0, 3, 7), "y": [1.0, 2.0, 3.0]})
    a = ar.from_xarray(x)
    if kwargs.get("w") == "y":
        kwargs = {"w": np.linspace(1, 2, 7)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = x.isel(y=[0, 2]).polyfit("x", 2, **kwargs)
        got = a.polyfit("x", 2, **kwargs)
    for k in want.data_vars:
        g = got[k].isel(y=[0, 2]) if "y" in got[k].dims else got[k]
        if k == "polyfit_covariance" and kwargs.get("cov") == "unscaled":
            # annoray's covariance runs along the other dimensions too.
            g = g.isel(y=0)
        close(g, want[k], rtol=1e-9)
    assert list(got.coords["degree"].values) == [2, 1, 0]
    lane = got["polyfit_coefficients"].isel(y=1).values
    assert np.isnan(lane).all() == (kwargs.get("skipna") is False)


def test_polyval_round_trip_with_units():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    t = ar.DataArray(15 - 6.5 * z + 0.1 * z**2, dims="z", attrs={"units": "degC"})
    t = t.assign_coords(z=ar.DataArray(z, dims="z", attrs={"units": "km"}))
    fit = t.polyfit("z", 2)
    np.testing.assert_allclose(fit["polyfit_coefficients"].values, [0.1, -6.5, 15.0], atol=1e-12)
    at = ar.DataArray([500.0, 2500.0], dims="h", attrs={"units": "m"})
    out = ar.polyval(at, fit["polyfit_coefficients"])
    assert out.unit == ar.Unit("degC")
    np.testing.assert_allclose(out.values, [11.775, -0.625])
    ds = ar.polyval(at, fit)
    np.testing.assert_allclose(ds["polyfit_coefficients"].values, out.values)


def test_polyval_matches_xarray_on_times():
    a, x = pair()
    a2, x2 = a.isel(y=0).swap_dims({"x": "t"}), x.isel(y=0).swap_dims({"x": "t"})
    got = ar.polyval(a2.coords["t"], a2.polyfit("t", 2)["polyfit_coefficients"])
    want = xr.polyval(x2.t, x2.polyfit("t", 2).polyfit_coefficients)
    np.testing.assert_allclose(got.values, want.values, rtol=1e-6)


def test_datasets():
    a, x = pair()
    ds = ar.Dataset({"v": a, "w": a * 2.0})
    xs = xr.Dataset({"v": x, "w": x * 2.0})
    close(ds.differentiate("x")["w"], xs.differentiate("x")["w"])
    close(ds.integrate("x")["w"], xs.integrate("x")["w"])
    close(ds.cumulative_integrate("x")["v"], xs.cumulative_integrate("x")["v"])
    close(
        ds.polyfit("x", 1)["w_polyfit_coefficients"],
        xs.polyfit("x", 1)["w_polyfit_coefficients"],
        rtol=1e-9,
    )
