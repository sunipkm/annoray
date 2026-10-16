"""xarray's conveniences: copy, astype, assign_attrs, pipe, renames, expand_dims, merge,
Python scalars, mapping methods, .loc, numpy reductions with axis=, where, *_like."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def build(m, units=None):
    attrs = {"k": "v"}
    da = m.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0], "lab": ("x", ["a", "b"])},
        name="a",
        attrs=attrs,
    )
    if units and m is ar:
        da = ar.DataArray(
            da.values,
            dims=da.dims,
            coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0]},
            name="a",
            attrs=attrs,
            units=units,
        )
    ds = m.Dataset({"a": da, "b": (("x",), np.array([1.0, 2.0]))}, attrs={"t": "g"})
    return da, ds


def same(a, b):
    """Same dims, values and coordinate names as the xarray result."""
    assert tuple(a.dims) == tuple(b.dims)
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))
    assert sorted(a.coords) == sorted(b.coords)


def test_copy_astype_assign_attrs_pipe():
    da, ds = build(ar)
    xda, _ = build(xr)
    c = da.copy(data=np.ones((2, 3)))
    assert c.attrs == {"k": "v"} and c.name == "a"
    np.testing.assert_array_equal(c.values, 1.0)
    with pytest.raises(ValueError, match="dimension"):
        da.copy(data=np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        da.copy(data=np.ones((3, 2)))
    same(da.astype("int32"), xda.astype("int32"))
    assert da.astype(np.float32).dtype == np.float32
    assert ds.astype("float32")["b"].dtype == np.float32
    # Values with no counterpart are errors (numpy: garbage).
    with pytest.raises(ValueError, match="NaN"):
        ar.DataArray([np.nan, 1.0], dims="x").astype(int)
    with pytest.raises(ValueError, match="out of range"):
        ar.DataArray([300], dims="x").astype("uint8")
    assert da.assign_attrs({"a": 1}, b="2").attrs["b"] == "2"
    assert ds.assign_attrs(z="q").attrs == {"t": "g", "z": "q"}
    assert float(da.pipe(lambda a, k: a * k, 2)[0, 1]) == 2.0
    assert float(da.pipe((lambda k, arr: arr * k, "arr"), 3)[0, 1]) == 3.0
    assert isinstance(ds.pipe(lambda d: d * 2), ar.Dataset)


def test_drop_vars_and_renames():
    da, ds = build(ar)
    xda, xds = build(xr)
    assert sorted(da.drop_vars("lab").coords) == sorted(xda.drop_vars("lab").coords)
    with pytest.raises(KeyError):
        da.drop_vars("zz")
    assert sorted(da.drop_vars(["zz", "x"], errors="ignore").coords) == ["lab", "y"]
    assert list(ds.drop_vars(["zz", "b"], errors="ignore")) == ["a"]
    for f in (
        lambda d: d.rename({"a": "c"}),
        lambda d: d.rename(x="xx", a="aa"),
        lambda d: d.rename_vars(a="c"),
        lambda d: d.rename_dims(x="xx"),
    ):
        r, xr_ = f(ds), f(xds)
        assert list(r.data_vars) == list(xr_.data_vars)
        assert sorted(r.coords) == sorted(xr_.coords)
        for k in r.data_vars:
            assert r[k].dims == xr_[k].dims
    # A swap.
    assert ds.rename_vars(a="b", b="a")["a"].dims == ("x",)
    with pytest.raises(ValueError):
        ds.rename_vars(a="b")


def test_expand_dims_and_merge():
    da, ds = build(ar)
    xda, xds = build(xr)
    for f in (
        lambda o: o.expand_dims("t"),
        lambda o: o.expand_dims(["t", "u"]),
        lambda o: o.expand_dims(t=3),
        lambda o: o.expand_dims({"t": [5, 6]}),
        lambda o: o.expand_dims({"t": 2}, axis=1),
    ):
        same(f(da), f(xda))
        for k in xds.data_vars:
            same(f(ds)[k], f(xds)[k])
    np.testing.assert_array_equal(da.expand_dims(t=[5, 6]).t.values, [5, 6])
    other = ar.Dataset({"c": (("x",), np.array([5.0, 6.0]))}, coords={"x": [10.0, 20.0]})
    m = ds.merge(other)
    assert sorted(m.data_vars) == ["a", "b", "c"]


def test_python_scalars():
    da, _ = build(ar)
    assert float(da[0, 1]) == 1.0
    assert int(da.astype(int)[1, 2]) == 5
    assert bool(da[0, 1]) and not bool(da[0, 0])
    assert da.item(4) == 4.0 and da[1, 0].item() == 3.0
    with pytest.raises(ValueError, match="ambiguous"):
        bool(da)
    with pytest.raises(TypeError, match="size-1"):
        float(da)
    with pytest.raises(TypeError, match="unsized"):
        len(da[0, 0])
    # Units: a plain number only for dimensionless data; item() gives a Quantity.
    m = ar.DataArray([2.5], dims="x", units="m")
    with pytest.raises(ar.UnitError):
        float(m)
    q = m.item()
    assert isinstance(q, ar.Quantity) and q == ar.Quantity(2.5, "m")
    assert float(ar.DataArray([5.0], dims="x", units="%")) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(ar.DataArray([0.0], dims="x", units="degC"))


def test_mapping_methods_and_loc():
    da, ds = build(ar)
    xda, xds = build(xr)
    assert [k for k, _ in ds.items()] == ["a", "b"]
    assert all(isinstance(v, ar.DataArray) for v in ds.values())
    assert ds.get("a").name == "a" and ds.get("zz", 5) == 5 and ds.get("zz") is None
    for key in (10.0, (slice(None), 2.0), (..., 2.0), {"y": 2.0}, slice(10.0, 20.0), [10.0]):
        same(da.loc[key], xda.loc[key])
    same(ds.loc[{"x": 10.0}]["a"], xds.loc[{"x": 10.0}]["a"])
    with pytest.raises(TypeError, match="immutable"):
        da.loc[{"x": 10.0}] = 1.0
    with pytest.raises(IndexError):
        da.loc[1.0, 2.0, 3.0]


def test_numpy_reductions_with_axis():
    da, ds = build(ar)
    xda, _ = build(xr)
    for f in (
        lambda a: np.sum(a, axis=0),
        lambda a: np.mean(a, axis=-1),
        lambda a: np.std(a, axis=0),
        lambda a: np.max(a),
        lambda a: np.min(a, axis=(0, 1)),
        lambda a: a.sum(axis=1),
        lambda a: np.prod(a, axis=1),
    ):
        same(f(da), f(xda))
    with pytest.raises(IndexError):
        da.sum(axis=2)
    with pytest.raises(ValueError, match="not both"):
        da.sum("x", axis=0)
    with pytest.raises(TypeError, match="out="):
        da.sum(out=np.zeros(3))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="dim="):
        ds.sum(axis=0)


def test_like_constructors():
    da, ds = build(ar)
    xda, xds = build(xr)
    same(ar.zeros_like(da), xr.zeros_like(xda))
    same(ar.full_like(da, 7.0), xr.full_like(xda, 7.0))
    assert ar.ones_like(ds).attrs == {"t": "g"}
    same(ar.ones_like(ds)["b"], xr.ones_like(xds)["b"])
    assert ar.full_like(da, 2, dtype="int16").dtype == np.int16
    # numpy truncates 0.5 to 0 in an integer array; here it is an error.
    with pytest.raises(ValueError, match="whole"):
        ar.full_like(da.astype(int), 0.5)
    m = ar.DataArray([1.0, 2.0], dims="x", units="m")
    assert ar.zeros_like(m).units == "m"
    np.testing.assert_allclose(ar.full_like(m, ar.Quantity(1, "km")).values, 1000.0)


def test_where_three_way():
    da, _ = build(ar)
    xda, _ = build(xr)
    for f in (
        lambda m, a: m.where(a > 2, a, -1.0),
        lambda m, a: m.where(a.isel(y=0) > 2, 1.0, a),
        lambda m, a: m.where(a > 2, a, a * 10),
    ):
        same(f(ar, da), f(xr, xda))
    r = ar.where(da > 2, da, -1.0)
    assert r.attrs == {"k": "v"} and r.name == "a"
    m = ar.DataArray([1.0, 2.0, 3.0], dims="x", units="m")
    km = ar.Quantity(1, "km")
    np.testing.assert_allclose(ar.where(m > ar.Quantity(1.5, "m"), m, km).values, [1000, 2, 3])
    with pytest.raises(ar.UnitError):
        ar.where(m > ar.Quantity(1.5, "m"), m, 5.0)
    assert np.isnan(ar.where(m > ar.Quantity(1.5, "m"), m, np.nan).values[0])


def test_where_method_other_and_callable():
    da, ds = build(ar)
    xda, xds = build(xr)
    same(da.where(lambda a: a > 2), xda.where(lambda a: a > 2))
    same(da.where(da > 2, -da), xda.where(xda > 2, -xda))
    same(ds.where(lambda d: d > 2)["a"], xds.where(lambda d: d > 2)["a"])
    m = ar.DataArray([1.0, 2.0, 3.0], dims="x", units="m")
    cond = m > ar.Quantity(1.5, "m")
    np.testing.assert_allclose(m.where(cond, ar.Quantity(1, "km")).values, [1000, 2, 3])
    np.testing.assert_allclose(m.where(cond, 0).values, [0, 2, 3])
    with pytest.raises(ar.UnitError):
        m.where(cond, 5.0)
    mds = ar.Dataset({"m": m})
    with pytest.raises(ar.UnitError):
        mds.where(mds > ar.Quantity(1.5, "m"), 5.0)
    np.testing.assert_allclose(
        mds.where(mds > ar.Quantity(1.5, "m"), ar.Quantity(1, "km"))["m"].values, [1000, 2, 3]
    )


def test_lazy_stays_lazy():
    ds = ar.open_dataset(FIXTURES / "nc" / "xr_nc4.nc")
    name = next(k for k, v in ds.data_vars.items() if v.is_lazy and v.dtype == np.float64)
    a = ds[name]
    assert a.astype("float32").is_lazy
    assert a.where(a > 0, 0.0).is_lazy
    assert ds.rename_vars({name: "renamed"})["renamed"].is_lazy
