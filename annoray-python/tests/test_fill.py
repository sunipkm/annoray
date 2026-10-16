"""ffill, bfill and interpolate_na, checked against xarray and scipy."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
interpolate = pytest.importorskip("scipy.interpolate")

NAN = np.nan
X = np.array([0.0, 1, 2.5, 3, 4.5, 5, 7, 8, 9.5, 10])


def pair(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((3, 10))
    v[rng.random((3, 10)) < 0.35] = NAN
    v[0, :2] = NAN
    v[1, -3:] = NAN
    x = xr.DataArray(v, dims=("y", "x"), coords={"x": X, "y": [1, 2, 3]}, attrs={"units": "m"})
    return ar.from_xarray(x), x


def close(a, x, rtol=1e-9):
    np.testing.assert_allclose(
        np.asarray(a.values, float),
        np.asarray(x.values, float),
        rtol=rtol,
        atol=1e-12,
        equal_nan=True,
    )
    assert tuple(a.dims) == tuple(x.dims)


@pytest.mark.parametrize(
    "method", ["linear", "nearest", "zero", "slinear", "cubic", "pchip", "akima"]
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_methods_match_xarray(method, seed):
    a, x = pair(seed)
    if method == "cubic" and (x.notnull().sum("x") < 4).any():
        pytest.skip("scipy needs four points for cubic")
    close(a.interpolate_na("x", method=method), x.interpolate_na("x", method=method))


@pytest.mark.parametrize("method", ["linear", "nearest", "zero", "slinear", "cubic"])
def test_extrapolation_matches_xarray(method):
    a, x = pair(4)
    kw = {"fill_value": "extrapolate"}
    close(a.interpolate_na("x", method=method, **kw), x.interpolate_na("x", method=method, **kw))


@pytest.mark.parametrize("method", ["pchip", "akima", "makima"])
def test_hermite_extrapolation_matches_scipy(method):
    y = np.array([NAN, 1.0, NAN, 4.0, 0.5, NAN, NAN, 3.0, 2.0, NAN])
    v = ~np.isnan(y)
    if method == "pchip":
        f = interpolate.PchipInterpolator(X[v], y[v], extrapolate=True)
    else:
        f = interpolate.Akima1DInterpolator(X[v], y[v], method=method, extrapolate=True)
    want = np.where(v, y, f(X))
    got = ar.DataArray(y, dims="x").assign_coords(x=X)
    got = got.interpolate_na("x", method=method, fill_value="extrapolate")
    np.testing.assert_allclose(got.values, want, rtol=1e-12)


def test_limit_and_use_coordinate():
    a, x = pair(5)
    close(a.interpolate_na("x", limit=1), x.interpolate_na("x", limit=1))
    close(
        a.interpolate_na("x", use_coordinate=False),
        x.interpolate_na("x", use_coordinate=False),
    )
    a2 = a.assign_coords(pos=ar.DataArray(X**2, dims="x"))
    x2 = x.assign_coords(pos=("x", X**2))
    close(
        a2.interpolate_na("x", use_coordinate="pos"), x2.interpolate_na("x", use_coordinate="pos")
    )
    assert a.interpolate_na("x").unit == ar.Unit("m")
    assert a.interpolate_na("x", keep_attrs=False).attrs == {}
    with pytest.raises(ValueError):
        a.interpolate_na("x", method="krogh")
    with pytest.raises(ValueError):
        a.interpolate_na()


@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("extrapolate", [False, True])
def test_max_gap_matches_xarray(gap, extrapolate):
    pytest.importorskip("bottleneck")
    y = np.array([NAN, 1.0, NAN, 4.0, 0.5, NAN, NAN, 3.0, 2.0, NAN])
    x = xr.DataArray(y, dims="x", coords={"x": X})
    kw = {"fill_value": "extrapolate"} if extrapolate else {}
    close(
        ar.from_xarray(x).interpolate_na("x", max_gap=gap, **kw),
        x.interpolate_na("x", max_gap=gap, **kw),
    )


def test_max_gap_on_times():
    pytest.importorskip("bottleneck")
    t = np.arange(6).astype("datetime64[D]").astype("datetime64[ns]")
    x = xr.DataArray([0.0, NAN, 2.0, NAN, NAN, 5.0], dims="t", coords={"t": t})
    a = ar.from_xarray(x)
    for g, same in [("2D", "2D"), ("3D", "3D"), ("2 d", "2D"), ("48h", "2D")]:
        close(a.interpolate_na("t", max_gap=g), x.interpolate_na("t", max_gap=same))
    for g in [np.timedelta64(2, "D"), ar.Quantity(48.0, "h")]:
        close(a.interpolate_na("t", max_gap=g), x.interpolate_na("t", max_gap="2D"))
    with pytest.raises(ValueError):
        a.interpolate_na("t", max_gap=2.0)


def test_ffill_bfill():
    pytest.importorskip("bottleneck")
    a, x = pair(6)
    close(a.ffill("x"), x.ffill("x"))
    close(a.bfill("x", limit=1), x.bfill("x", limit=1))
    ds, xs = ar.Dataset({"v": a}), xr.Dataset({"v": x})
    close(ds.ffill("x", limit=2)["v"], xs.ffill("x", limit=2)["v"])


def test_ffill_bfill_without_bottleneck():
    y = [NAN, 1.0, NAN, NAN, 4.0, NAN]
    a = ar.DataArray(y, dims="x")
    np.testing.assert_allclose(a.ffill("x").values, [NAN, 1, 1, 1, 4, 4], equal_nan=True)
    np.testing.assert_allclose(a.bfill("x", limit=1).values, [1, 1, NAN, 4, 4, NAN], equal_nan=True)


def test_datasets():
    a, x = pair(7)
    ds, xs = ar.Dataset({"v": a, "w": a * 2.0}), xr.Dataset({"v": x, "w": x * 2.0})
    close(ds.interpolate_na("x", method="pchip")["w"], xs.interpolate_na("x", method="pchip")["w"])
