"""Tier 3: pickling and copying, isel/sel drop=, file options, decode_cf, append mode,
GroupBy map/reduce/quantile, rolling over several dimensions, drop_duplicates,
reindex(method=), dt extras, date_range calendars, align(exclude=), concat/merge options,
accessors."""

from __future__ import annotations

import copy
import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")

FILE = Path(__file__).resolve().parents[2] / "fixtures" / "nc" / "xr_nc4.nc"


def build(m):
    da = m.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0, 4.0], "lab": ("x", ["a", "b", "a"])},
        name="a",
        attrs={"k": "v"},
    )
    return da, m.Dataset({"a": da, "b": (("x",), np.array([1.0, 2.0, 2.0]))}, attrs={"t": "g"})


def same(a, b):
    assert tuple(a.dims) == tuple(b.dims)
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_pickle_and_copy_keep_everything():
    ds = ar.open_dataset(FILE)
    ds["temp"].encoding.update(zlib=True)
    for obj in (ds, ds["temp"], ar.DataArray([1.0], dims="x", units="km", name="d")):
        back = pickle.loads(pickle.dumps(obj))
        ar.testing.assert_identical(back, obj)
        ar.testing.assert_identical(copy.deepcopy(obj), obj)
        ar.testing.assert_identical(copy.copy(obj), obj)
    assert pickle.loads(pickle.dumps(ds))["temp"].encoding.get("zlib") is True


def test_drop_and_missing_dims():
    da, ds = build(ar)
    xda, xds = build(xr)
    assert sorted(da.isel(x=0, drop=True).coords) == sorted(xda.isel(x=0, drop=True).coords)
    assert sorted(ds.sel(x=10.0, drop=True).coords) == sorted(xds.sel(x=10.0, drop=True).coords)
    assert da.isel(z=0, missing_dims="ignore").dims == da.dims
    with pytest.warns(UserWarning):
        da.isel(z=0, missing_dims="warn")
    with pytest.raises(ValueError, match="not a dimension"):
        da.isel(z=0)


def test_plumbing():
    da, ds = build(ar)
    with ar.open_dataset(FILE) as f:
        assert "temp" in f
    ds.close()
    assert sorted(ds.variables) == ["a", "b", "lab", "x", "y"]
    assert list(da.xindexes) == ["x", "y"]
    buf = io.StringIO()
    ds.info(buf=buf)
    assert "dimensions:" in buf.getvalue() and ":t = " in buf.getvalue()
    assert ds._repr_html_().startswith("<pre>")
    same(da.real, da)
    assert float(da.imag.max()) == 0.0
    same(da.conj(), da)


def test_open_options(tmp_path):
    assert list(ar.open_dataset(FILE, chunks={}, engine="h5netcdf").data_vars)
    assert "precip" not in ar.open_dataset(FILE, drop_variables="precip")
    raw = ar.open_dataset(FILE, decode_cf=False)
    x = xr.open_dataset(FILE, decode_cf=False)
    assert raw["time"].dtype == x["time"].dtype
    decoded = ar.decode_cf(raw)
    ar.testing.assert_equal(decoded["temp"], ar.open_dataset(FILE)["temp"])
    assert ar.open_dataset(FILE, use_cftime=True)["time"].values.dtype == object
    with pytest.raises(ValueError, match="engine"):
        ar.open_dataset(FILE, engine="cfgrib")
    with pytest.raises(TypeError, match="unexpected"):
        ar.open_dataset(FILE, nonsense=1)
    assert len(ar.load_dataset(FILE).data_vars) == 6
    # Append mode: variables added, the file kept intact.
    _, ds = build(ar)
    path = tmp_path / "f.nc"
    ds.to_netcdf(path)
    ds.rename_vars(a="c")[["c"]].to_netcdf(path, mode="a")
    assert sorted(ar.open_dataset(path).data_vars) == ["a", "b", "c"]
    assert sorted(xr.open_dataset(path).data_vars) == ["a", "b", "c"]
    nested = ar.open_mfdataset([FILE, FILE], combine="nested", concat_dim="run")
    assert nested.sizes["run"] == 2
    pre = ar.open_mfdataset(
        [FILE, FILE], combine="nested", concat_dim="run", preprocess=lambda d: d[["temp"]]
    )
    assert list(pre.data_vars) == ["temp"]


def test_groupby_and_rolling():
    da, ds = build(ar)
    xda, _ = build(xr)
    same(
        da.groupby("lab").map(lambda g: g - g.mean()),
        xda.groupby("lab").map(lambda g: g - g.mean()),
    )
    same(da.groupby("lab").reduce(np.max), xda.groupby("lab").reduce(np.max))
    same(da.groupby("lab").quantile(0.5), xda.groupby("lab").quantile(0.5))
    same(
        ds.groupby("lab").map(lambda g: g * 2)["a"],
        build(xr)[1].groupby("lab").map(lambda g: g * 2)["a"],
    )
    windows: list[dict[str, Any]] = [
        {"x": 2, "y": 2},
        {"x": 2, "y": 3, "center": True},
        {"x": 2, "y": 2, "min_periods": 1},
    ]
    for kw in windows:
        for op in ("mean", "sum", "max", "count"):
            same(getattr(da.rolling(**kw), op)(), getattr(xda.rolling(**kw), op)())
    same(
        da.rolling(x=2, y=2).construct(x="wx", y="wy"),
        xda.rolling(x=2, y=2).construct(x="wx", y="wy"),
    )


def test_drop_duplicates_and_reindex_method():
    da, _ = build(ar)
    xda, _ = build(xr)
    dup = da.assign_coords(x=[1.0, 1.0, 2.0])
    xdup = xda.assign_coords(x=[1.0, 1.0, 2.0])
    for keep in ("first", "last", False):
        same(dup.drop_duplicates("x", keep=keep), xdup.drop_duplicates("x", keep=keep))
    options: list[dict[str, Any]] = [
        {"method": "nearest"},
        {"method": "pad"},
        {"method": "backfill"},
        {"method": "nearest", "tolerance": 2},
        {"fill_value": -1.0},
    ]
    for kw in options:
        same(
            da.drop_vars("lab").reindex(x=[11.0, 25.0, 40.0], **kw),
            xda.drop_vars("lab").reindex(x=[11.0, 25.0, 40.0], **kw),
        )


def test_dt_extras():
    t = pd.date_range("2000-01-01 06:30", periods=3, freq="17h").values
    a, x = ar.DataArray(t, dims="t"), xr.DataArray(t, dims="t")
    assert (
        a.dt.strftime("%Y-%m-%d %H").values.tolist() == x.dt.strftime("%Y-%m-%d %H").values.tolist()
    )
    for op in ("floor", "ceil", "round"):
        same(getattr(a.dt, op)("D"), getattr(x.dt, op)("D"))
    d = np.array([90061, -5, 3600 * 30], dtype="timedelta64[s]")
    a, x = ar.DataArray(d, dims="t"), xr.DataArray(d, dims="t")
    for f in ("days", "seconds", "microseconds", "nanoseconds"):
        same(getattr(a.dt, f), getattr(x.dt, f))
    same(a.dt.total_seconds(), x.dt.total_seconds())


def test_date_ranges():
    for cal in ("noleap", "360_day", "all_leap", "julian"):
        r = ar.date_range("2000-02-27", periods=5, calendar=cal)
        x = xr.date_range("2000-02-27", periods=5, calendar=cal, use_cftime=True)
        assert [str(t) for t in r] == [str(t) for t in x]
    r = ar.date_range("2000-01-15", end="2001-01-01", freq="MS", calendar="noleap")
    x = xr.date_range("2000-01-15", end="2001-01-01", freq="MS", calendar="noleap")
    assert [str(t) for t in r] == [str(t) for t in x]
    assert len(ar.cftime_range("2000", periods=3)) == 3
    np.testing.assert_array_equal(
        ar.date_range("2000", "2000-01-10"), xr.date_range("2000", "2000-01-10").values
    )


def test_combining_options():
    da, ds = build(ar)
    xda, _ = build(xr)
    shifted = da.assign_coords(y=[3.0, 4.0, 5.0, 6.0])
    r = ar.align(da, shifted, join="outer", exclude=["y"])
    x = xr.align(xda, xda.assign_coords(y=[3.0, 4.0, 5.0, 6.0]), join="outer", exclude=["y"])
    for a, b in zip(r, x, strict=True):
        same(a, b)
    same(
        ar.concat([da, da], dim=pd.Index([5, 6], name="run")),
        xr.concat([xda, xda], dim=pd.Index([5, 6], name="run")),
    )
    assert ar.concat([da, da], dim=pd.Index([5, 6], name="run"))["run"].values.tolist() == [5, 6]
    c = ar.concat([ds, ds], dim="x", data_vars="all", join="outer")
    assert c.sizes["x"] == 6
    with pytest.raises(ValueError):
        ar.merge([ds, ds.assign(a=ds["a"] + 1)], compat="identical")
    assert len(ar.merge([ds, ds], compat="override").data_vars) == 2
    assert ar.get_options()["keep_attrs"] is True
    assert ar.unify_chunks(da, ds) == (da, ds)


def test_accessors():
    @ar.register_dataarray_accessor("geo")
    class Geo:
        def __init__(self, obj):
            self._obj = obj
            self.calls = 0

        def doubled(self):
            self.calls += 1
            return self._obj * 2

    @ar.register_dataset_accessor("summary")
    class Summary:
        def __init__(self, obj):
            self._obj = obj

        def names(self):
            return sorted(self._obj.data_vars)

    da, ds = build(ar)
    assert float(da.geo.doubled().max()) == 22.0
    da.geo.doubled()
    assert da.geo.calls == 2, "the accessor is cached per object"
    assert ds.summary.names() == ["a", "b"]
    with pytest.warns(Warning, match="overriding"):
        ar.register_dataarray_accessor("mean")(Geo)
