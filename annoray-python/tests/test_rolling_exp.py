"""rolling_exp: against xarray (numbagg), numbagg directly for min_weight, units, and
numerical robustness where numbagg's E[x²] − E[x]² form fails."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
numbagg = pytest.importorskip("numbagg")

rng = np.random.default_rng(0)
V = rng.normal(size=(3, 40))
V[0, 5:9] = np.nan
V[2, 0] = np.nan
W = V * 2 + V**2

WINDOWS = [
    ("span", 5, 2 / 6),
    ("com", 3, 1 / 4),
    ("halflife", 4, 1 - 0.5 ** (1 / 4)),
    ("alpha", 0.3, 0.3),
    ("span", 1, 1.0),
]
SINGLE = {
    "mean": numbagg.move_exp_nanmean,
    "sum": numbagg.move_exp_nansum,
    "std": numbagg.move_exp_nanstd,
    "var": numbagg.move_exp_nanvar,
}


def close(r, e):
    np.testing.assert_array_equal(np.isnan(r), np.isnan(e))
    np.testing.assert_allclose(r, e, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("window_type", "window", "alpha"), WINDOWS)
def test_matches_xarray_and_numbagg(window_type, window, alpha):
    a, x = ar.DataArray(V, dims=("s", "t")), xr.DataArray(V, dims=("s", "t"))
    aw, xw = ar.DataArray(W, dims=("s", "t")), xr.DataArray(W, dims=("s", "t"))
    ra = a.rolling_exp(t=window, window_type=window_type)
    rx = x.rolling_exp(t=window, window_type=window_type)
    for op, f in SINGLE.items():
        close(getattr(ra, op)().values, getattr(rx, op)().values)
        # xarray drops min_weight (it lands among the window keywords); numbagg's own rule:
        rm = a.rolling_exp(t=window, window_type=window_type, min_weight=0.5)
        close(getattr(rm, op)().values, f(V, alpha=alpha, min_weight=0.5, axis=-1))
    close(ra.cov(aw).values, rx.cov(xw).values)
    close(ra.corr(aw).values, rx.corr(xw).values)
    rm = a.rolling_exp(t=window, window_type=window_type, min_weight=0.5)
    close(rm.cov(aw).values, numbagg.move_exp_nancov(V, W, alpha=alpha, min_weight=0.5, axis=-1))
    close(rm.corr(aw).values, numbagg.move_exp_nancorr(V, W, alpha=alpha, min_weight=0.5, axis=-1))


def test_dataset_types_and_units():
    ds = ar.Dataset(
        {"a": ar.DataArray(V, dims=("s", "t"), units="m"), "c": ar.DataArray([1.0, 2.0], dims="u")},
        attrs={"k": "v"},
    )
    r = ds.rolling_exp(t=5).mean()
    close(r["a"].values, xr.DataArray(V, dims=("s", "t")).rolling_exp(t=5).mean().values)
    assert r["a"].units == "m" and r["c"].values.tolist() == [1.0, 2.0] and r.attrs == {"k": "v"}
    m = ds["a"].rolling_exp(t=5)
    assert ar.Unit(str(m.var().units)) == ar.Unit("m2")
    assert m.std().units == "m" and m.corr(ds["a"]).units is None
    t = ar.DataArray([270.0, 280.0, 275.0], dims="t", units="degC")
    assert t.rolling_exp(t=2).std().units == "K"
    with pytest.raises(ar.UnitError):
        t.rolling_exp(t=2).sum()
    f32 = ar.DataArray(V.astype(np.float32), dims=("s", "t"))
    assert f32.rolling_exp(t=5).mean().dtype == np.float32
    with pytest.raises(ValueError, match="span"):
        f32.rolling_exp(t=0.5)
    with pytest.raises(ValueError, match="one dimension"):
        f32.rolling_exp(t=2, s=2)


@pytest.mark.parametrize("offset", [1e6, 1e9, 1e12])
def test_stable_where_numbagg_cancels(offset):
    data = rng.normal(size=200) * 1e-3 + offset
    # The variance is shift-invariant: of the data as stored, shifted back to zero (an
    # exact subtraction), the textbook form has nothing to cancel.
    exact = numbagg.move_exp_nanvar(data - offset, alpha=2 / 21)
    ours = ar.DataArray(data, dims="t").rolling_exp(t=20).var().values
    theirs = numbagg.move_exp_nanvar(data, alpha=2 / 21)
    m = ~np.isnan(exact)
    assert np.max(np.abs(ours[m] - exact[m]) / exact[m]) < 1e-12
    assert np.max(np.abs(theirs[m] - exact[m]) / exact[m]) > 1.0
