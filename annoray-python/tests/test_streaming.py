"""The xarray-convenience surfaces on lazily opened data: the same results as on loaded
data, and lazy (nothing read) wherever the operation allows it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import annoray as ar

FILE = Path(__file__).resolve().parents[2] / "fixtures" / "nc" / "xr_nc4.nc"
K = ar.Quantity(280, "K")


def T(d):
    return d["temp"]


def P(d):
    return d["precip"]


# (operation, whether the result stays lazy without reading the source)
CASES = {
    "astype": (lambda d: T(d).astype("float64"), True),
    "astype same type": (lambda d: d[["temp", "precip"]].astype("float32")["temp"], True),
    "where scalar": (lambda d: T(d).where(T(d) > K, 0.0), True),
    "where quantity": (lambda d: T(d).where(T(d) > K, ar.Quantity(1, "degC")), True),
    "where array": (lambda d: T(d).where(T(d) > K, T(d) * 2), True),
    "ar.where": (lambda d: ar.where(T(d) > K, T(d), K), True),
    "fillna array": (lambda d: T(d).fillna(T(d).mean("time")), True),
    "combine_first": (lambda d: T(d).combine_first(T(d) * 2), True),
    "broadcast_like": (lambda d: d["flag"].broadcast_like(T(d)), True),
    "to_dataarray": (lambda d: d[["temp"]].to_dataarray(), True),
    "rename": (lambda d: d.rename(temp="t")["t"], True),
    "expand_dims sized": (lambda d: T(d).expand_dims(e=3), True),
    "head": (lambda d: T(d).head(time=1), True),
    "loc": (lambda d: T(d).loc[..., d.lon.values[1]], True),
    "cov": (lambda d: ar.cov(T(d), P(d), dim="time"), False),
    "corr": (lambda d: ar.corr(T(d), P(d)), False),
    "sum axis": (lambda d: T(d).sum(axis=0), False),
    "pad": (lambda d: T(d).pad(time=1), False),
    "rank": (lambda d: T(d).rank("time"), False),
    "reduce": (lambda d: T(d).reduce(np.nanmax, "time"), False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_lazy_matches_loaded(name):
    f, stays_lazy = CASES[name]
    lazy = ar.open_dataset(FILE)
    source = lazy["temp"]
    result = f(lazy)
    if stays_lazy:
        assert result.is_lazy, "result should stay lazy"
        assert source.is_lazy, "the source should not have been read"
    loaded = f(ar.open_dataset(FILE).load())
    ar.testing.assert_identical(result, loaded)


def test_reductions_and_equality_stream():
    lazy = ar.open_dataset(FILE)
    source = lazy["temp"]
    for f in (
        lambda d: ar.cov(T(d), P(d), dim="time"),
        lambda d: T(d).equals(T(d) * 1),
        lambda d: d.identical(d.copy()),
        lambda d: ar.broadcast(T(d), d["flag"])[1].mean(),
    ):
        f(lazy)
    assert source.is_lazy, "streamed: the whole array was never held"
    other = ar.open_dataset(FILE)
    assert T(lazy).equals(T(other)) and lazy.identical(other)
    assert not T(lazy).equals(T(other).where(T(other) < K))
