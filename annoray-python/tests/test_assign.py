"""Assignment: ds["v"] = …, coords, names, live attrs (written through to the source
variable), and memory safety of numpy views across replacements."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import annoray as ar


def build():
    da = ar.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0]},
        name="a",
        attrs={"k": "v"},
    )
    return da, ar.Dataset({"a": da}, attrs={"t": "g"})


def test_attrs_are_live_dicts():
    da, ds = build()
    da.attrs["q"] = "1"
    assert da.attrs == {"k": "v", "q": "1"}
    da.attrs.update(r="2")
    del da.attrs["k"]
    assert da.attrs.pop("q") == "1"
    assert da.attrs.setdefault("s", "3") == "3"
    assert da.attrs == {"r": "2", "s": "3"}
    da.attrs = {"only": "this"}
    assert da.attrs == {"only": "this"}
    ds.attrs["u"] = "x"
    ds.attrs |= {"w": "y"}
    assert ds.attrs == {"t": "g", "u": "x", "w": "y"}
    # Plain-dict behaviour.
    assert isinstance(ds.attrs, dict)
    assert json.loads(json.dumps(ds.attrs)) == {"t": "g", "u": "x", "w": "y"}
    assert type(pickle.loads(pickle.dumps(ds.attrs))) is dict
    detached = ds.attrs.copy()
    detached["z"] = "1"
    assert "z" not in ds.attrs
    ds.attrs.clear()
    assert ds.attrs == {}


def test_attrs_write_through_to_the_source_variable():
    _, ds = build()
    ds["a"].attrs["w"] = "1"
    ds.a.attrs["x"] = "2"
    ds.data_vars["a"].attrs["y"] = "3"
    dict(ds.items())["a"].attrs["z"] = "4"
    ds.coords["x"].attrs["units_note"] = "c"
    assert {"w", "x", "y", "z"} <= set(ds["a"].attrs)
    assert ds["x"].attrs == {"units_note": "c"}
    # A coordinate of a data array.
    da = ds["a"]
    da.x.attrs["n"] = "1"
    assert da.coords["x"].attrs["n"] == "1"
    # A handle to a replaced variable no longer writes through.
    old = ds["a"]
    ds["a"] = ds["a"] * 2
    old.attrs["stale"] = "1"
    assert "stale" not in ds["a"].attrs


def test_dataset_setitem_delitem_update():
    _, ds = build()
    ds["b"] = ds["a"] * 2
    assert ds["b"].name == "b" and list(ds.data_vars) == ["a", "b"]
    ds["c"] = ("x", np.array([1.0, 2.0]))
    ds["d"] = (("x", "y"), np.ones((2, 3)), {"note": "n"})
    assert ds["d"].attrs == {"note": "n"}
    ds["a"] = ds["a"] + 1
    np.testing.assert_array_equal(ds["a"].values[0], [1.0, 2.0, 3.0])
    # Refused: wrong length. The dataset is unchanged.
    before = list(ds.data_vars)
    with pytest.raises(ValueError, match="length"):
        ds["bad"] = ("x", np.zeros(5))
    assert list(ds.data_vars) == before
    # A coordinate's name replaces the coordinate.
    ds["x"] = ("x", np.array([1.0, 2.0]))
    np.testing.assert_array_equal(ds.x.values, [1.0, 2.0])
    del ds["c"]
    assert "c" not in ds
    ds.update({"e": ds["a"] - 1})
    other = ar.Dataset({"f": ("x", np.array([0.0, 0.0]))}, coords={"x": [1.0, 2.0]})
    assert ds.update(other) is ds
    assert {"e", "f"} <= set(ds.data_vars)
    with pytest.raises(KeyError):
        del ds["nope"]


def test_coords_and_names():
    da, ds = build()
    ds.coords["lab"] = ("x", ["p", "q"])
    assert list(ds["lab"].values) == ["p", "q"]
    del ds.coords["lab"]
    assert "lab" not in ds.coords
    with pytest.raises(KeyError):
        del ds.coords["a"]  # a data variable
    da.coords["lab"] = ("x", ["p", "q"])
    da["y"] = [5.0, 6.0, 7.0]
    np.testing.assert_array_equal(da.y.values, [5.0, 6.0, 7.0])
    del da["lab"]
    assert "lab" not in da.coords
    with pytest.raises(ValueError):
        da.coords["z"] = ("z", [1.0])  # not a dimension of the data
    da.name = "renamed"
    assert da.name == "renamed"
    da.name = None
    assert da.name is None


def test_values_stay_immutable():
    da, _ = build()
    with pytest.raises(TypeError, match="immutable"):
        da[0, 0] = 5.0  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="immutable"):
        da.loc[{"x": 10.0}] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        da.values[0, 0] = 5.0


def test_views_survive_replacement():
    pytest.importorskip("xarray")
    da, ds = build()
    view = da.coords["x"].values
    x = ds.to_xarray()
    kept = x["a"].values.copy()
    for _ in range(3):
        ds["a"] = ds["a"] * 0
        ds.coords["x"] = ("x", np.array([0.0, 0.0]))
        del ds["a"]
        ds["a"] = ("x", np.zeros(2))
    np.testing.assert_array_equal(x["a"].values, kept)
    np.testing.assert_array_equal(view, [10.0, 20.0])
    xa = da.to_xarray()
    da["x"] = [0.0, 0.0]
    del da["y"]
    np.testing.assert_array_equal(xa.x.values, [10.0, 20.0])
    np.testing.assert_array_equal(xa.y.values, [1.0, 2.0, 3.0])


ATTRS = {
    "short": np.int16(-3),
    "byte": np.uint8(7),
    "single": np.float32(0.1),
    "ints": np.array([1, 2, 3], dtype=np.int32),
    "doubles": np.array([0.5, 1.5]),
    "text": "hello",
}


def same_attr(a, b):
    assert np.asarray(a).dtype == np.asarray(b).dtype, (a, b)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_attribute_edits_keep_netcdf_types(tmp_path):
    xr = pytest.importorskip("xarray")
    src, out = tmp_path / "src.nc", tmp_path / "out.nc"
    xr.Dataset({"v": (("x",), np.arange(3.0), dict(ATTRS))}, attrs=dict(ATTRS)).to_netcdf(src)
    ds = ar.open_dataset(src)
    ds.attrs["added"] = np.int16(5)
    del ds.attrs["text"]
    ds["v"].attrs.update(note="edited")
    ds.to_netcdf(out)
    back = xr.open_dataset(out)
    for k, a in ATTRS.items():
        same_attr(back["v"].attrs[k], a)
        if k != "text":
            same_attr(back.attrs[k], a)
    assert "text" not in back.attrs
    same_attr(back.attrs["added"], np.int16(5))
    assert back["v"].attrs["note"] == "edited"
