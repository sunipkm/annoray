"""numpy ufuncs on DataArrays and apply_ufunc, checked against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")
RNG = np.random.default_rng(5)


def pair(units=None, shape=(3, 4)):
    v = RNG.random(shape) + 0.5
    x = xr.DataArray(
        v,
        dims=("y", "x")[: len(shape)],
        coords={
            "y": np.arange(shape[0]) * 10.0,
            **({"x": np.arange(shape[1])} if len(shape) > 1 else {}),
        },
        name="v",
        attrs={} if units is None else {"units": units},
    )
    return ar.from_xarray(x), x


def same(a, x):
    np.testing.assert_allclose(np.asarray(a.values, float), np.asarray(x.values, float), rtol=1e-12)
    assert tuple(a.dims) == tuple(x.dims)


@pytest.mark.parametrize(
    "name", ["sqrt", "exp", "log", "sin", "floor", "negative", "absolute", "square", "tanh", "sign"]
)
def test_ufuncs_without_units(name):
    a, x = pair()
    f = getattr(np, name)
    out = f(a)
    assert isinstance(out, ar.DataArray)
    same(out, f(x))
    assert list(out.coords) == list(x.coords)


def test_ufuncs_with_units():
    a, _ = pair(units="m2")
    r = np.sqrt(a)
    assert r.unit == ar.Unit("m")
    assert (a**2).unit is not None and (a**2).unit.is_convertible(ar.Unit("m4"))
    with pytest.raises(ar.UnitError):
        np.exp(a)
    d, _ = pair(units="degree")
    np.testing.assert_allclose(np.sin(d).values, np.sin(np.deg2rad(d.values)))
    # Inverse functions give radians for dimensionless input with a unit; plain data stays plain.
    one, _ = pair(units="1")
    assert np.arcsin(one * 0.5).unit == ar.Unit("rad")
    assert np.arcsin(np.sin(d)).unit is None
    c, _ = pair(units="degC")
    with pytest.raises(ar.UnitError):
        _ = -c
    m, _ = pair(units="m")
    assert (-m).unit == ar.Unit("m") and abs(m).unit == ar.Unit("m")
    s, _ = pair(units="s")
    assert np.maximum(m, m).unit == ar.Unit("m")
    assert np.multiply(m, s).unit.is_convertible(ar.Unit("m s"))
    # Not unit-aware: fine without a unit, refused with one.
    plain, plain_x = pair()
    same(np.deg2rad(plain), np.deg2rad(plain_x))
    with pytest.raises(TypeError):
        np.deg2rad(m)


def test_apply_ufunc_elementwise_and_reductions():
    a, x = pair(units="m")
    same(ar.apply_ufunc(np.square, a), xr.apply_ufunc(np.square, x))
    got = ar.apply_ufunc(np.mean, a, input_core_dims=[["x"]], kwargs={"axis": -1})
    want = xr.apply_ufunc(np.mean, x, input_core_dims=[["x"]], kwargs={"axis": -1})
    same(got, want)
    np.testing.assert_array_equal(got.coords["y"].values, want.coords["y"].values)
    assert got.unit is None and got.attrs == {}
    kept = ar.apply_ufunc(np.mean, a, input_core_dims=[["x"]], kwargs={"axis": -1}, keep_attrs=True)
    assert kept.unit == ar.Unit("m")
    u = ar.apply_ufunc(np.square, a, output_units=["m2"])
    assert u.unit == ar.Unit("m2")


def test_apply_ufunc_broadcasting_and_alignment():
    a, x = pair()
    bx = xr.DataArray(RNG.random(4), dims="x", coords={"x": np.arange(4)})
    b = ar.from_xarray(bx)
    same(ar.apply_ufunc(np.add, a, b), xr.apply_ufunc(np.add, x, bx))
    # Alignment: inner join on labels.
    c = a.isel(x=slice(1, 3))
    cx = x.isel(x=slice(1, 3))
    same(ar.apply_ufunc(np.add, a, c, join="inner"), xr.apply_ufunc(np.add, x, cx, join="inner"))
    with pytest.raises(ValueError):
        ar.apply_ufunc(np.add, a, c)
    # Plain numpy arguments pass through.
    same(ar.apply_ufunc(np.add, a, 2.0), xr.apply_ufunc(np.add, x, 2.0))


def test_apply_ufunc_core_dims_vectorize_and_outputs():
    a, x = pair()

    def norm(v):
        return np.sqrt((v**2).sum())

    kw = {"input_core_dims": [["x"]], "vectorize": True}
    same(ar.apply_ufunc(norm, a, **kw), xr.apply_ufunc(norm, x, **kw))

    def both(v):
        return v.min(axis=-1), v.max(axis=-1)

    kw = {"input_core_dims": [["x"]], "output_core_dims": [[], []]}
    lo, hi = ar.apply_ufunc(both, a, **kw)
    xlo, xhi = xr.apply_ufunc(both, x, **kw)
    same(lo, xlo)
    same(hi, xhi)

    def diffs(v):
        return np.diff(v, axis=-1)

    kw = {"input_core_dims": [["x"]], "output_core_dims": [["x"]], "exclude_dims": {"x"}}
    same(ar.apply_ufunc(diffs, a, **kw), xr.apply_ufunc(diffs, x, **kw))
    with pytest.raises(ValueError):
        ar.apply_ufunc(np.mean, a, input_core_dims=[["nope"]])


def test_apply_ufunc_datasets():
    a, x = pair()
    ds, xs = ar.Dataset({"p": a, "q": a * 2.0}), xr.Dataset({"p": x, "q": x * 2.0})
    got = ar.apply_ufunc(np.mean, ds, input_core_dims=[["x"]], kwargs={"axis": -1})
    want = xr.apply_ufunc(np.mean, xs, input_core_dims=[["x"]], kwargs={"axis": -1})
    for k in ["p", "q"]:
        same(got[k], want[k])


def test_rename_mapping():
    a, x = pair()
    same(a.rename({"x": "lon"}), x.rename({"x": "lon"}))
    assert "lon" in a.rename(x="lon").coords
    assert a.rename("w").name == "w"
