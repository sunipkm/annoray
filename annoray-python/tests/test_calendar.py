"""convert_calendar against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
pytest.importorskip("cftime")


def times(values):
    return [str(t).replace("T", " ")[:19] for t in np.asarray(values)]


def pair(start, n, calendar="standard"):
    xt = xr.date_range(
        start, periods=n, freq="D", calendar=calendar, use_cftime=calendar != "standard"
    )
    x = xr.DataArray(
        np.arange(float(n)), dims="time", coords={"time": xt}, name="v", attrs={"k": "v"}
    )
    return ar.from_xarray(x), x


CASES = [
    ("2000-02-27", 5, "standard", "noleap", {}),
    ("2000-02-27", 5, "standard", "all_leap", {}),
    ("2000-02-27", 5, "standard", "julian", {}),
    ("2000-02-27", 5, "noleap", "standard", {}),
    ("2000-02-27", 5, "noleap", "standard", {"missing": -1.0}),
    ("2000-02-27", 5, "standard", "360_day", {"align_on": "date"}),
    ("2000-01-29", 5, "standard", "360_day", {"align_on": "date"}),
    ("2000-01-01", 366, "standard", "360_day", {"align_on": "year"}),
    ("2000-02-28", 4, "360_day", "standard", {"align_on": "date"}),
    ("2000-02-28", 4, "360_day", "standard", {"align_on": "year"}),
    ("2001-01-01", 360, "360_day", "noleap", {"align_on": "year"}),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[2]}->{c[3]}{c[4]}" for c in CASES])
def test_matches_xarray(case):
    start, n, source, target, kw = case
    a, x = pair(start, n, source)
    r, e = a.convert_calendar(target, **kw), x.convert_calendar(target, **kw)
    assert times(r.time.values) == times(e.time.values)
    np.testing.assert_array_equal(r.values, e.values)
    assert r.attrs == e.attrs and r.name == e.name


def test_rules_and_datasets():
    a, _ = pair("2000-02-27", 5)
    with pytest.raises(ValueError, match="align_on"):
        a.convert_calendar("360_day")
    with pytest.raises(ValueError, match="use_cftime"):
        a.convert_calendar("noleap", use_cftime=False)
    assert a.convert_calendar("standard", use_cftime=True).time.values.dtype == object
    ds = ar.Dataset({"v": a, "w": a * 2, "c": ("other", np.arange(2.0))})
    out = ds.convert_calendar("noleap")
    assert out.sizes["time"] == 4 and out["c"].sizes == {"other": 2}
    np.testing.assert_array_equal(out["w"].values, [0, 2, 6, 8])
    filled = out.convert_calendar("standard", missing=-1.0)
    np.testing.assert_array_equal(filled["w"].values, [0, 2, -1, 6, 8])
    np.testing.assert_array_equal(filled["c"].values, [0, 1])
    with pytest.raises(TypeError, match="needs times"):
        ar.DataArray([1.0], dims="time", coords={"time": [1.0]}).convert_calendar("noleap")
