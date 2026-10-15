"""The native annoray objects against xarray: the same operations must give the same values
and coordinates (`assert_allclose` on `to_xarray()`), units must be tracked and checked, and
the numpy views annoray hands out must be read-only."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import annoray as ar

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nc"
REDUCTIONS = ["sum", "prod", "mean", "median", "std", "var", "min", "max", "count"]


def check(got, expected):
    """Values and coordinates as xarray computes them (attributes, units aside)."""
    xr.testing.assert_allclose(got.to_xarray(), expected)


def pair(**kw):
    """The same random array (with NaNs) as annoray and xarray objects."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, 4, 3))
    data[1, 2, 0] = np.nan
    data[4, :, 1] = np.nan
    t = pd.date_range("2001-01-30", periods=6, freq="7D")
    coords = {"time": t, "y": [10.0, 20.0, 30.0, 40.0], "x": ["a", "b", "c"]}
    x = xr.DataArray(data, dims=("time", "y", "x"), coords=coords, name="v")
    r = ar.DataArray(data, dims=("time", "y", "x"), coords=coords, name="v", **kw)
    return r, x


# ----- construction and properties -----


def test_properties():
    r, x = pair(units="K")
    assert r.dims == x.dims and r.shape == x.shape and r.sizes == dict(x.sizes)
    assert r.dtype == x.dtype and r.ndim == 3 and r.size == x.size and len(r) == 6
    assert r.units == "K" and r.name == "v"
    assert list(r.coords) == list(x.coords)
    np.testing.assert_array_equal(r.time.values, x.time.values)
    np.testing.assert_array_equal(r.x.values, x.x.values)
    check(r, x)


def test_units_from_attrs_and_quantities():
    a = ar.DataArray([1.0, 2.0], dims="x", attrs={"units": "m s-1", "long_name": "speed"})
    assert a.units is not None and "units" not in a.attrs and a.attrs["long_name"] == "speed"
    q = ar.DataArray(2.0 * ar.Unit("km"))  # a scalar quantity: a 0-d array
    assert q.units == "km" and q.ndim == 0 and float(q.values) == 2.0
    with pytest.warns(UserWarning, match="kept as an attribute"):
        b = ar.DataArray([1.0], dims="x", attrs={"units": "not a unit at all"})
    assert b.units is None and b.attrs["units"] == "not a unit at all"
    with pytest.raises(ar.UnitError):
        ar.DataArray([1.0], dims="x", units="furlongs per fortnight squared!")


def test_values_are_read_only_views():
    r, _ = pair()
    v = r.values
    assert np.shares_memory(v, r.values), "no copy for in-memory numbers"
    with pytest.raises(ValueError):
        v[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        v.setflags(write=True)
    del r  # the view keeps the data alive
    assert np.isfinite(v[0, 0, 0])
    assert np.asarray(pair()[0]).shape == (6, 4, 3)


def test_dtypes_round_trip():
    t = np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]")
    d = np.array([3600, "NaT"], dtype="timedelta64[s]").astype("timedelta64[ns]")
    for data in [t, d, np.array(["a", "bcd"]), np.array([True, False]), np.arange(2, dtype="u2")]:
        r = ar.DataArray(data, dims="x")
        np.testing.assert_array_equal(r.values, data)
        assert r.dtype == (data.dtype if data.dtype.kind != "U" else np.dtype(object))
    import cftime

    c = np.array([cftime.DatetimeNoLeap(2001, 2, 28), cftime.DatetimeNoLeap(2001, 3, 1)])
    r = ar.DataArray(c, dims="x")
    assert list(r.values) == list(c)


# ----- arithmetic and units -----


def test_arithmetic_matches_xarray():
    r, x = pair()
    check(r + r, x + x)
    check(r * 2.0 - 1.0, x * 2.0 - 1.0)
    check(2.0 / (r + 10.0), 2.0 / (x + 10.0))
    # Inner join on shared labels, as xarray does.
    check(
        r.isel(y=slice(0, 3)) + r.isel(y=slice(1, 4)), x.isel(y=slice(0, 3)) + x.isel(y=slice(1, 4))
    )


def test_units_are_checked():
    a = ar.DataArray([1.0, 2.0], dims="x", units="m")
    s = ar.DataArray([2.0, 4.0], dims="x", units="s")
    assert (a / s).units == "m s-1"
    assert (a * a).units in {"m2", "m^2", "m²"}
    with pytest.raises(ar.UnitError):
        _ = a + s
    with pytest.raises(ar.UnitError):
        _ = a + 1.0  # a bare number is dimensionless
    with pytest.raises(ar.UnitError, match="convert one first"):
        _ = a + ar.Quantity(1.0, "km")
    np.testing.assert_allclose(
        (a + ar.Quantity(1.0, "km").__class__(1000.0, "m")).values, [1001, 1002]
    )
    np.testing.assert_allclose(a.to("cm").values, [100.0, 200.0])
    c = ar.DataArray([0.0, 100.0], dims="x", units="degC")
    np.testing.assert_allclose(c.to("K").values, [273.15, 373.15])
    np.testing.assert_allclose(c.to("degF").values, [32.0, 212.0])


def test_pint_quantities():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    a = ar.DataArray(np.array([1.0, 2.0]) * u.meter, dims="x", coords={"x": [0.0, 1000.0]})
    assert a.units == "m"
    np.testing.assert_allclose((a + 1.0 * u.m).values, [2.0, 3.0])
    assert (a * (2.0 * u.s)).units is not None
    with pytest.raises(ar.UnitError):
        _ = a + 50.0 * u.cm
    with pytest.raises(ar.UnitError):
        _ = a + 1.0 * u.s


def test_quantity_labels():
    a = ar.DataArray(
        [1.0, 2.0], dims="x", coords={"x": ar.DataArray([0.0, 1000.0], dims="x", units="m")}
    )
    assert a.sel(x=ar.Quantity(1.0, "km")).values == 2.0
    with pytest.raises(ar.UnitError):
        a.sel(x=ar.Quantity(1.0, "s"))


# ----- selection -----

ISEL = [
    {"time": 0},
    {"time": -1, "x": 1},
    {"time": slice(1, 5, 2)},
    {"y": slice(None, None, -1)},
    {"time": [0, 3, 1]},
    {"y": np.array([True, False, True, True])},
    {"time": [5, 0], "x": [2, 0]},
]


@pytest.mark.parametrize("ix", ISEL, ids=str)
def test_isel_matches(ix):
    r, x = pair()
    check(r.isel(ix), x.isel(ix))
    check(r.isel(**ix), x.isel(**ix))


SEL = [
    ({"y": 20.0}, {}),
    ({"y": slice(15.0, 35.0)}, {}),
    ({"x": "b"}, {}),
    ({"x": ["c", "a"]}, {}),
    ({"time": "2001-02"}, {}),
    ({"time": slice("2001-02-01", "2001-02-28")}, {}),
    ({"time": np.datetime64("2001-02-06")}, {}),
    ({"y": 24.0}, {"method": "nearest"}),
    ({"y": 24.0}, {"method": "ffill"}),
    ({"y": 24.0}, {"method": "bfill"}),
    ({"y": [11.0, 39.0]}, {"method": "nearest"}),
]


@pytest.mark.parametrize(("ix", "kw"), SEL, ids=str)
def test_sel_matches(ix, kw):
    r, x = pair()
    check(r.sel(ix, **kw), x.sel(ix, **kw))


def test_transpose_matches():
    r, x = pair()
    check(r.transpose("x", "time", "y"), x.transpose("x", "time", "y"))
    check(r.transpose("y", ...), x.transpose("y", ...))
    check(r.T, x.T)
    ds = r.to_dataset()
    xr.testing.assert_allclose(
        ds.transpose("x", ...).to_xarray(), x.to_dataset().transpose("x", ...)
    )


def test_getitem_matches():
    r, x = pair()
    for key in [0, (1, slice(1, 3)), (Ellipsis, 0), (slice(None), [0, 2]), {"x": 1}]:
        check(r[key], x[key])


def test_selection_errors():
    r, _ = pair()
    with pytest.raises(KeyError):
        r.sel(y=25.0)
    with pytest.raises(ValueError):
        r.isel(nope=0)
    with pytest.raises(IndexError):
        r.isel(time=10)
    with pytest.raises(AttributeError):
        r.nope  # noqa: B018


# ----- reductions -----


@pytest.mark.parametrize("op", REDUCTIONS)
@pytest.mark.parametrize("dim", [None, "time", ["y", "x"]])
def test_reductions_match(op, dim):
    r, x = pair()
    check(getattr(r, op)(dim), getattr(x, op)(dim))


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reductions_without_skipna(op):
    r, x = pair()
    check(getattr(r, op)("time", skipna=False), getattr(x, op)("time", skipna=False))


def test_reduction_options():
    r, x = pair()
    check(r.std("time", ddof=1), x.std("time", ddof=1))
    check(r.sum("time", min_count=6), x.sum("time", min_count=6))
    check(r.isel(x=0).argmax("y"), x.isel(x=0).argmax("y"))
    # An all-NaN slice has no position; both raise.
    with pytest.raises(ValueError):
        x.argmax("y")
    with pytest.raises(ValueError):
        r.argmax("y")


def test_reduction_units():
    a = ar.DataArray([1.0, 2.0, 4.0], dims="x", units="m")
    assert a.mean().units == "m" and a.std().units == "m"
    assert a.var().units in {"m2", "m^2", "m²"}
    assert a.count().units is None


# ----- group-by, resample, rolling -----


@pytest.mark.parametrize("op", ["mean", "sum", "max", "count"])
def test_groupby_matches(op):
    r, x = pair()
    check(getattr(r.groupby("time.month"), op)(), getattr(x.groupby("time.month"), op)())
    check(getattr(r.groupby("x"), op)(), getattr(x.groupby("x"), op)())


def test_groupby_iteration_and_arithmetic():
    r, x = pair()
    assert len(r.groupby("time.month")) == len(x.groupby("time.month"))
    for (kr, gr), (kx, gx) in zip(r.groupby("time.month"), x.groupby("time.month"), strict=True):
        assert kr == kx
        check(gr, gx)
    clim_r = r.groupby("time.month").mean()
    clim_x = x.groupby("time.month").mean()
    check(r.groupby("time.month") - clim_r, x.groupby("time.month") - clim_x)


def test_groupby_bins_matches():
    r, x = pair()
    got = r.groupby_bins("y", [0, 25, 50]).sum().to_xarray()
    expected = x.groupby_bins("y", [0, 25, 50]).sum()
    # Bin labels are their text ("(0, 25]"), not pandas Interval objects.
    expected = expected.assign_coords(y_bins=[str(i) for i in expected.y_bins.values])
    xr.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("freq", ["14D", "MS", "ME"])
def test_resample_matches(freq):
    r, x = pair()
    check(r.resample(time=freq).mean(), x.resample(time=freq).mean())


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("op", ["mean", "sum", "max", "std"])
def test_rolling_matches(center, op):
    r, x = pair()
    check(
        getattr(r.rolling(time=3, center=center), op)(),
        getattr(x.rolling(time=3, center=center), op)(),
    )
    check(
        getattr(r.rolling(time=3, min_periods=1), op)(),
        getattr(x.rolling(time=3, min_periods=1), op)(),
    )


@pytest.mark.parametrize("part", ["year", "month", "day", "hour", "dayofyear", "dayofweek"])
def test_dt_matches(part):
    r, x = pair()
    np.testing.assert_array_equal(getattr(r.time.dt, part).values, getattr(x.time.dt, part).values)


# ----- missing values, interpolation, combining -----


def test_missing_values_match():
    r, x = pair()
    check(r.isnull(), x.isnull())
    check(r.fillna(0.0), x.fillna(0.0))
    check(r.where(r.notnull(), -1.0), x.where(x.notnull(), -1.0))
    check(r.dropna("time", how="all"), x.dropna("time", how="all"))
    check(r.dropna("time", how="any"), x.dropna("time", how="any"))
    check(r.reindex(y=[10.0, 15.0, 40.0]), x.reindex(y=[10.0, 15.0, 40.0]))


def test_interp_matches():
    r, x = pair()
    at = [12.5, 20.0, 37.0]
    check(r.interp(y=at), x.interp(y=at))
    check(r.interp(y=at, method="nearest"), x.interp(y=at, method="nearest"))


def test_combining_matches():
    r, x = pair()
    check(
        ar.concat([r.isel(time=slice(0, 2)), r.isel(time=slice(2, 6))], "time"),
        xr.concat([x.isel(time=slice(0, 2)), x.isel(time=slice(2, 6))], "time"),
    )
    a, b = ar.align(r.isel(y=slice(0, 3)), r.isel(y=slice(1, 4)), join="outer")
    xa, xb = xr.align(x.isel(y=slice(0, 3)), x.isel(y=slice(1, 4)), join="outer")
    check(a, xa)
    check(b, xb)
    m = ar.merge([r.rename("a"), (r * 2.0).rename("b")])
    xm = xr.merge([x.rename("a"), (x * 2.0).rename("b")])
    xr.testing.assert_allclose(m.to_xarray(), xm)


# ----- datasets and files -----


def test_dataset_construction_and_access():
    ds = ar.Dataset(
        {"a": ("x", [1.0, 2.0], {"units": "m"}), "b": ("x", np.array([3, 4]))},
        coords={"x": [5, 6]},
        attrs={"title": "t"},
    )
    assert ds.keys() == ["a", "b"] and list(ds) == ["a", "b"] and len(ds) == 2
    assert "a" in ds and "x" in ds and "nope" not in ds
    assert ds.a.units == "m" and ds["a"].units == "m"
    assert ds.attrs == {"title": "t"} and ds.sizes == {"x": 2}
    assert ds[["a"]].keys() == ["a"]
    assert ds.assign(c=ds.a * 2.0).c.units == "m"
    assert ds.drop_vars("b").keys() == ["a"]
    assert ds.set_coords("b").keys() == ["a"]
    with pytest.raises(KeyError):
        ds["nope"]
    xr.testing.assert_identical(
        ds.to_xarray(),
        xr.Dataset(
            {"a": ("x", [1.0, 2.0], {"units": "m"}), "b": ("x", np.array([3, 4]))},
            coords={"x": [5, 6]},
            attrs={"title": "t"},
        ),
    )


@pytest.mark.parametrize("name", ["xr_nc3.nc", "xr_nc4.nc", "raw_nc4.nc"])
def test_open_dataset_matches_xarray(name):
    ours = ar.open_dataset(FIXTURES / name)
    theirs = xr.open_dataset(FIXTURES / name)
    assert set(ours.keys()) == set(theirs.data_vars)
    for k in theirs.variables:
        a, b = ours[str(k)], theirs[k]
        assert a.dims == b.dims
        if b.dtype.kind in "fiub":
            np.testing.assert_allclose(a.values, b.values, equal_nan=True)
        elif b.dtype.kind == "S":
            # Deviation: text without `_Encoding` stays bytes in xarray; annoray decodes it
            # (as UTF-8, the netCDF default).
            assert list(np.ravel(a.values)) == [s.decode() for s in np.ravel(b.values)]
        else:
            assert list(np.ravel(a.values)) == list(np.ravel(b.values))
        if "units" in b.attrs and a.units is not None:
            assert ar.DataArray([1.0], dims="x", units=b.attrs["units"]).units == a.units


def test_lazy_reads_and_load():
    ds = ar.open_dataset(FIXTURES / "xr_nc4.nc")
    t = ds.temp
    assert t.is_lazy
    loaded = t.load()
    assert not loaded.is_lazy
    np.testing.assert_array_equal(t.values, loaded.values)
    np.testing.assert_array_equal(t.isel(time=1).values, loaded.values[1])


def test_masked_noleap_time_opens():
    # xarray cannot decode this variable (OverflowError in cftime); annoray decodes it.
    ds = ar.open_dataset(FIXTURES / "nat.nc")
    v = ds.t_noleap.values
    assert v[1] is None or v[1] != v[1] or v[1] is not None


def test_to_netcdf_round_trip(tmp_path):
    r, x = pair(units="K")
    ds = r.to_dataset()
    ds.to_netcdf(tmp_path / "a.nc")
    back = xr.open_dataset(tmp_path / "a.nc")
    xr.testing.assert_allclose(back.v, x)
    assert back.v.attrs["units"] == "K"
    again = ar.open_dataset(tmp_path / "a.nc")
    assert again.v.units == "K"
    ar.to_netcdf(ds, tmp_path / "b.nc", format="NETCDF3_64BIT")
    xr.testing.assert_allclose(xr.open_dataset(tmp_path / "b.nc").v, x)


def test_open_mfdataset(tmp_path):
    r, x = pair()
    ds = r.to_dataset()
    ds.isel(time=slice(0, 3)).to_netcdf(tmp_path / "p1.nc")
    ds.isel(time=slice(3, 6)).to_netcdf(tmp_path / "p2.nc")
    both = ar.open_mfdataset(str(tmp_path / "p*.nc"))
    check(both.v, x)
    nested = ar.open_mfdataset([tmp_path / "p1.nc", tmp_path / "p2.nc"], concat_dim="time")
    check(nested.v, x)


def test_xarray_round_trip():
    x = xr.open_dataset(FIXTURES / "xr_nc4.nc").load()
    r = ar.from_xarray(x)
    assert r.temp.units == "K"
    back = r.to_xarray()
    for k in x.variables:
        assert back[k].dims == x[k].dims
        if x[k].dtype.kind in "fiubM":
            np.testing.assert_array_equal(back[k].values, x[k].values)
    assert back.temp.attrs["units"] == "K"


def test_threads_share_objects():
    r, x = pair()
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda i: r.isel(time=i % 6).mean().values, range(200)))
    expect = [float(x.isel(time=i % 6).mean()) for i in range(200)]
    np.testing.assert_allclose(out, expect)


# ----- disk to disk -----


@pytest.fixture
def big_file(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.normal(size=(40, 30, 20))
    data[5, 3, :] = np.nan
    x = xr.Dataset(
        {"v": (("t", "y", "x"), data, {"units": "K"})},
        coords={"t": np.arange(40.0), "y": np.arange(30.0), "x": np.arange(20.0)},
    )
    path = tmp_path / "big.nc"
    x.to_netcdf(path, encoding={"v": {"zlib": True, "chunksizes": (5, 30, 20)}})
    return path, x


def test_lazy_arithmetic_streams_and_matches(big_file, tmp_path):
    path, x = big_file
    ds = ar.open_dataset(path)
    expr = (ds.v * 2.0 + ds.v.isel(t=0)).to("degC")
    assert expr.is_lazy, "nothing computed until read or written"
    xr.testing.assert_allclose(
        expr.to_xarray().drop_attrs(), ((x.v * 2.0 + x.v.isel(t=0)) - 273.15).drop_attrs()
    )
    ar.Dataset({"w": expr}).to_netcdf(tmp_path / "w.nc")
    back = xr.open_dataset(tmp_path / "w.nc")
    np.testing.assert_allclose(back.w, (x.v * 2.0 + x.v.isel(t=0)) - 273.15)


@pytest.mark.parametrize("op", ["mean", "sum", "std", "var", "min", "max"])
@pytest.mark.parametrize("dim", ["t", "y", ["t", "x"], None])
def test_streamed_reductions_match(big_file, op, dim):
    path, x = big_file
    old = ar.stream_bytes()
    ar.set_stream_bytes(8 * 30 * 20 * 3)  # three time steps at a time
    try:
        ds = ar.open_dataset(path)
        assert ds.v.is_lazy
        check(getattr(ds.v, op)(dim), getattr(x.v, op)(dim))
    finally:
        ar.set_stream_bytes(old)
    assert ar.stream_bytes() == old


def test_write_encoding(big_file, tmp_path):
    path, x = big_file
    ds = ar.open_dataset(path)
    out = tmp_path / "enc.nc"
    ds.to_netcdf(out, encoding={"v": {"zlib": True, "complevel": 7, "chunksizes": (10, 15, 20)}})
    back = xr.open_dataset(out)
    assert back.v.encoding["zlib"] and back.v.encoding["complevel"] == 7
    assert back.v.encoding["chunksizes"] == (10, 15, 20)
    xr.testing.assert_allclose(back.v, x.v)
    with pytest.raises(KeyError):
        ds.to_netcdf(tmp_path / "no.nc", encoding={"nope": {"zlib": True}})


# ----- DataTree -----


def test_datatree_matches_xarray():
    ours = ar.open_datatree(FIXTURES / "tree.nc")
    theirs = xr.open_datatree(FIXTURES / "tree.nc")
    # Depth first here, breadth first in xarray: the same nodes.
    assert sorted(ours.paths) == sorted(n.path for n in theirs.subtree)
    for node in theirs.subtree:
        got = ours.to_dataset(node.path).to_xarray()
        xr.testing.assert_allclose(
            got.drop_attrs(deep=True), node.to_dataset().load().drop_attrs(deep=True)
        )
    assert set(ours) == set(theirs.children)
    assert "obs" in ours and "nope" not in ours
    with pytest.raises(KeyError):
        ours["nope"]


def test_datatree_build_write_and_convert(tmp_path):
    root = ar.Dataset(coords={"x": [1.0, 2.0, 3.0]}, attrs={"title": "tree"})
    a = ar.Dataset({"t": ("x", [1.0, 2.0, 3.0], {"units": "K"})})
    b = ar.Dataset({"p": ("x", [5.0, 6.0, 7.0], {"units": "Pa"})})
    tree = ar.DataTree.from_dict({"/": root, "/a": a, "/a/b": b})
    assert tree.paths == ["/", "/a", "/a/b"]
    # Inherited coordinates.
    assert (
        "x" in tree.to_dataset("/a/b").coords
        and "x" not in tree.to_dataset("/a/b", inherit=False).coords
    )
    assert tree["a"]["b"].dataset["p"].units == "Pa"
    tree.to_netcdf(tmp_path / "t.nc")
    back = ar.open_datatree(tmp_path / "t.nc")
    assert back.paths == tree.paths
    assert back.to_dataset("/a").t.units == "K"
    np.testing.assert_array_equal(back.to_dataset("/a/b").p.values, [5.0, 6.0, 7.0])
    x = back.to_xarray()
    assert isinstance(x, xr.DataTree) and x["a/b"].p.attrs["units"] == "Pa"
    again = ar.from_xarray(x)
    assert isinstance(again, ar.DataTree) and again.paths == tree.paths


# ----- Unit and Quantity -----


def test_units_are_parsed_once_and_compose():
    km, s = ar.Unit("km"), ar.Unit("s")
    assert str(km / s) == "km s-1" and str(km**2) == "km2"
    assert km / s == ar.Unit("1000 m/s")  # equal whatever the spelling
    assert hash(km) == hash(ar.Unit("km")) and km != ar.Unit("m")
    assert km.is_convertible("m") and not km.is_convertible(s)
    assert ar.Unit("%").dimensionless and not km.dimensionless
    with pytest.raises(ar.UnitError):
        ar.Unit("not a unit!!")


def test_quantity_arithmetic_and_conversion():
    d = 2.5 * ar.Unit("km")
    assert isinstance(d, ar.Quantity) and d.magnitude == 2.5 and d.units == "km"
    assert f"{d:.1f}" == "2.5 km" and repr(d) == 'Quantity(2.5, "km")'
    v = d / ar.Quantity(30, "min")
    assert v.units == "km min-1"
    assert abs(v.to("m/s").magnitude - 2500 / 1800) < 1e-12
    assert (d**2).units == "km2" and (-d).magnitude == -2.5 and abs(-d) == d
    with pytest.raises(ar.UnitError, match="convert one first"):
        _ = d + ar.Quantity(1, "m")
    assert (d + ar.Quantity(1, "m").to("km")).magnitude == pytest.approx(2.501)
    with pytest.raises(ar.UnitError):
        _ = d + 1.0  # a plain number is dimensionless
    assert float(ar.Quantity(5, "%")) == pytest.approx(0.05)
    with pytest.raises(ar.UnitError):
        float(d)
    # Temperatures: absolute minus absolute is a difference.
    diff = ar.Quantity(20, "degC") - ar.Quantity(10, "degC")
    assert diff.magnitude == pytest.approx(10.0)
    assert ar.Quantity(20, "degC").to("degF").magnitude == pytest.approx(68.0)


def test_quantity_comparisons_convert():
    assert ar.Quantity(1, "km") == ar.Quantity(1000, "m")
    assert ar.Quantity(1, "km") > ar.Quantity(999, "m")
    assert ar.Quantity(1, "km") != ar.Quantity(1, "s")
    with pytest.raises(ar.UnitError):
        _ = ar.Quantity(1, "km") < ar.Quantity(1, "s")


def test_quantities_with_data_arrays():
    km = ar.Unit("km")
    x = ar.DataArray([0.0, 1000.0], dims="x", units=ar.Unit("m"))
    da = ar.DataArray([1.0, 2.0], dims="x", coords={"x": x}, units=km)
    assert da.unit == km and da.units == "km"
    np.testing.assert_allclose((da + 1 * km).values, [2.0, 3.0])
    assert (da * ar.Quantity(2, "s")).units == "km s"
    np.testing.assert_allclose(da.to(ar.Unit("m")).values, [1000.0, 2000.0])
    assert da.sel(x=1 * km).values == 2.0
    with pytest.raises(ar.UnitError):
        _ = da + ar.Quantity(1, "s")


# ----- named dimensions -----


def test_dimensions_compose_and_name():
    from annoray.dims import Energy, Force, Length, Mass, Time, Velocity

    assert Length / Time == Velocity.dimension
    assert (Force * Length) == Energy.dimension  # torque and energy: one dimension
    assert repr(Velocity.dimension) == "Dimension(length=1, time=-1)"
    assert str(Velocity.dimension) == "m s-1"

    class Jerk(ar.dims.Dim, dimension=Length / Time**3):
        """Rate of change of acceleration."""

    assert str(Jerk.dimension) == "m s-3" and ar.dims.name_of(Jerk.dimension) == "Jerk"
    made = ar.dimension("Snap", Length / Time**4)  # the run-time spelling
    assert str(made.dimension) == "m s-4" and isinstance(made, ar.dims.DimensionType)
    speed = ar.dimension("Speed", Length / Time)
    assert speed.dimension == Velocity.dimension
    assert ar.Dimension(mass=1, length=2, time=-2) == Energy.dimension
    assert ar.Unit("kJ").dimension == Energy.dimension and Mass.dimension.exponents == {"mass": 1}
    assert (1 / Time) == ar.dims.Frequency.dimension


def test_dimensioned_quantities():
    from annoray.dims import Length, Time, Velocity

    d = Length(2.5, "km")
    assert isinstance(d, ar.Quantity) and isinstance(d, Length) and not isinstance(d, Time)
    with pytest.raises(ar.UnitError, match="expected Length"):
        Length(1.0, "s")
    v = d / Time(30.0, "min")
    assert isinstance(v, Velocity)  # however it was made
    assert Velocity.check(v) is v
    with pytest.raises(ar.UnitError):
        Velocity.check(d)
    da = ar.DataArray([1.0, 2.0], dims="x", units="m/s")
    assert isinstance(da, Velocity) and da.dimension == Velocity.dimension
    assert ar.DataArray([1.0], dims="x").dimension is None
    assert not isinstance(3.0, Length)


def test_checked_functions():
    from typing import Annotated

    from annoray.dims import Energy, Length, Time, Velocity, checked

    @checked
    def speed(
        d: ar.Quantity[Length], t: ar.Quantity[Time], label: str = ""
    ) -> ar.Quantity[Velocity]:
        return d / t

    # Bare names work at run time too (type checkers prefer `Quantity[...]`).
    @checked
    def bare(d: Length) -> Length:  # pyright: ignore[reportInvalidTypeForm]
        return d

    with pytest.raises(ar.UnitError, match="argument 'd'"):
        bare(Time(1, "s"))  # pyright: ignore[reportArgumentType]  (on purpose)

    assert isinstance(speed(Length(1, "km"), Time(1, "h")), Velocity)
    with pytest.raises(ar.UnitError, match="argument 'd'"):
        speed(Time(1, "h"), Length(1, "km"))  # pyright: ignore[reportArgumentType]  (on purpose)

    @checked
    def wrong(d: Annotated[ar.Quantity, Length]) -> Energy:  # pyright: ignore[reportInvalidTypeForm]
        return d  # pyright: ignore[reportReturnType]  (wrong on purpose)

    with pytest.raises(ar.UnitError, match="result"):
        wrong(Length(1, "m"))
    with pytest.raises(ar.UnitError, match="argument 'd'"):
        wrong(Time(1, "s"))


def test_log_interpolation():
    energy = np.array([1.0, 10.0, 100.0])
    sigma = ar.DataArray(energy**2, dims="e", coords={"e": energy}, units="cm2")
    out = sigma.interp(e=[3.0, 50.0], method="loglog")
    np.testing.assert_allclose(out.values, [9.0, 2500.0])
    assert out.units == "cm2"
    z = np.array([0.0, 10.0, 30.0])
    n = ar.DataArray(np.exp(-z / 10), dims="z", coords={"z": z}, units="cm-3")
    np.testing.assert_allclose(n.interp(z=[5.0], method="log_value").values, [np.exp(-0.5)])
    p = ar.DataArray(np.array([0.0, 2.0]), dims="p", coords={"p": [1.0, 100.0]})
    np.testing.assert_allclose(p.interp(p=[10.0], method="log_coord").values, [1.0])
    with pytest.raises(ValueError, match="log_coord"):
        p.interp(p=[10.0], method="cubic")  # pyright: ignore[reportArgumentType]


def test_interp_extrapolation():
    x = np.array([0.0, 1.0, 2.0])
    a = ar.DataArray(np.array([0.0, 10.0, 20.0]), dims="x", coords={"x": x}, units="m")
    assert np.isnan(a.interp(x=[3.0]).values).all()
    np.testing.assert_allclose(a.interp(x=[-1.0, 3.0], extrapolate="extend").values, [-10, 30])
    np.testing.assert_allclose(a.interp(x=[-1.0, 3.0], extrapolate="edge").values, [0, 20])
    both = a.interp(x=[-1.0, 3.0], extrapolate=("edge", "extend"))
    np.testing.assert_allclose(both.values, [0, 30])
    e = np.array([10.0, 100.0])
    s = ar.DataArray(e**-0.8, dims="e", coords={"e": e})
    out = s.interp(e=[1000.0], method="loglog", extrapolate=(None, "extend"))
    np.testing.assert_allclose(out.values, [1000.0**-0.8])
    with pytest.raises(ValueError, match="extrapolate"):
        a.interp(x=[3.0], extrapolate="clamp")  # pyright: ignore[reportArgumentType]


def test_interp_power_law_extrapolation():
    e = np.array([10.0, 100.0, 1000.0])
    s = ar.DataArray(np.array([3e-16, 2e-16, 9e-17]), dims="e", coords={"e": e}, units="cm2")
    out = s.interp(e=[5.0, 2000.0], method="loglog", extrapolate=("edge", -1.7))
    np.testing.assert_allclose(out.values, [3e-16, 9e-17 * 0.5**1.7])
    assert out.units == "cm2"
