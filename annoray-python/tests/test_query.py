"""query and argsort against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def build(m):
    return m.Dataset(
        {"a": ("x", [1.0, 5.0, 3.0]), "b": ("y", [10.0, 20.0])},
        coords={"x": [0, 1, 2], "y": [5, 6], "lab": ("x", ["p", "q", "r"])},
        attrs={"t": "g"},
    )


QUERIES = [
    {"x": "a > 2"},
    {"x": "a > 2 & x < 2"},
    {"x": "(a > 2) and (x < 2)"},
    {"x": "a > 2 | x == 0"},
    {"x": "not (a > 2)"},
    {"x": "lab == 'q'"},
    {"x": "lab in ['p', 'r']"},
    {"x": "lab not in ['p']"},
    {"x": "1 < a < 5"},
    {"x": "a * 2 - 1 >= 5"},
    {"x": "a > 2", "y": "b < 15"},
]


@pytest.mark.parametrize("q", QUERIES, ids=[str(q) for q in QUERIES])
def test_matches_xarray(q):
    r, x = build(ar).query(q), build(xr).query(q)
    assert dict(r.sizes) == dict(x.sizes)
    for k in x.variables:
        np.testing.assert_array_equal(np.asarray(r[k].values), np.asarray(x[k].values))


def test_data_array_query_and_extras():
    ds, xds = build(ar), build(xr)
    np.testing.assert_array_equal(ds.a.query(x="a > 2").values, xds.a.query(x="a > 2").values)
    np.testing.assert_array_equal(ds.a.query(x="x > 0").values, [5.0, 3.0])
    assert ds.query(x="lab.str.startswith('q')")["x"].values.tolist() == [1]
    t = ar.Dataset(
        {"v": ("time", [1.0, 2.0, 3.0])},
        coords={"time": ar.date_range("2000-01-30", 3, "D")},
    )
    assert t.query(time="time.dt.month == 2")["v"].values.tolist() == [3.0]
    with pytest.raises(ValueError, match="dimension"):
        ds.query(z="a > 1")
    assert ds.query(z="a > 1", missing_dims="ignore").sizes == ds.sizes
    with pytest.raises(ValueError, match="along 'x'"):
        ds.query(x="b > 2")
    with pytest.raises(TypeError, match="booleans"):
        ds.query(x="a + 1")


def test_units_and_safety():
    m = ar.Dataset({"temp": ar.DataArray([270.0, 290.0], dims="x", units="K")})
    assert m.query(x="temp > Q(280, 'K')")["temp"].values.tolist() == [290.0]
    assert m.query(x="temp > Q(5, 'degC')")["temp"].values.tolist() == [290.0]
    with pytest.raises(ar.UnitError):
        m.query(x="temp > 280")
    for bad in (
        "temp.to_netcdf('x.nc')",
        "__import__('os')",
        "temp._set_attrs({})",
        "[t for t in temp]",
    ):
        with pytest.raises((ValueError, NameError)):
            m.query(x=bad)


def test_argsort():
    vals = [[3.0, np.nan, 1.0], [2.0, 0.0, 5.0]]
    a = ar.DataArray(
        vals, dims=("x", "y"), coords={"x": [1, 2]}, attrs={"k": 1}, name="n", units="m"
    )
    x = xr.DataArray(vals, dims=("x", "y"), coords={"x": [1, 2]}, attrs={"k": 1}, name="n")
    for r, e in (
        (a.argsort(), x.argsort()),
        (a.argsort(axis=0), x.argsort(axis=0)),
        (a.argsort(kind="stable"), x.argsort(kind="stable")),
    ):
        assert r.values.tolist() == e.values.tolist() and r.dims == e.dims
    s = a.argsort()
    assert s.units is None and s.name == "n" and list(s.coords) == ["x"] and s.attrs["k"] == 1
