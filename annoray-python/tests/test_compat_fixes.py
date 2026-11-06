"""Regression tests for the compatibility fixes glow-rs needed: shared buffers, `units` in
`attrs`, plain arrays in arithmetic and selection, dimension-named variables, unit parsing and
registration, scalar `interp`, `clip` with a bare number, `copy=False`, optional matplotlib."""

# Plain numpy arrays reach annoray's operators, which the stubs of numpy do not tell pyright.
# pyright: reportAttributeAccessIssue=false, reportUnusedExpression=false, reportOperatorIssue=false
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

import copy
import gc
import os
import subprocess
import sys
import textwrap
import warnings
import weakref
from typing import Any

import numpy as np
import pytest

import annoray as ar


def sample() -> ar.Dataset:
    return ar.Dataset(
        {"v": ("x", np.arange(3.0), {"units": "km"}), "s": ("x", np.array(["a", "b", "a"]))},
        coords={"x": ("x", np.arange(3.0), {"units": "m", "axis": "X"})},
        attrs={"title": "t"},
    )


# ----- item 1: shared buffers -----


def test_children_share_one_buffer():
    ds = ar.Dataset({"v": (("x", "y"), np.ones((4, 5)))}, coords={"x": np.arange(4.0)})
    assert np.shares_memory(ds["v"].values, ds["v"].values)
    assert np.shares_memory(ds["v"].values, ds.data_vars["v"].values)
    assert np.shares_memory(ds["v"].values, ds.variables["v"].values)
    assert np.shares_memory(ds["v"].values, ds.copy(deep=False)["v"].values)
    assert np.shares_memory(ds["x"].values, ds.coords["x"].values)
    assert np.shares_memory(ds["x"].values, ds["v"].coords["x"].values)


def test_deep_copy_owns_its_buffers():
    ds = ar.Dataset({"v": ("x", np.ones(4))}, coords={"x": np.arange(4.0)})
    deep = ds.copy(deep=True)
    assert not np.shares_memory(ds["v"].values, deep["v"].values)
    assert not np.shares_memory(ds["x"].values, deep["x"].values)
    da = ds["v"]
    assert np.shares_memory(da.values, da.copy(deep=False).values)
    assert not np.shares_memory(da.values, da.copy(deep=True).values)
    assert np.shares_memory(da.values, copy.copy(da).values)
    assert not np.shares_memory(da.values, copy.deepcopy(da).values)


@pytest.mark.parametrize("deep", [False, True])
def test_copies_are_independent_containers(deep):
    ds = sample()
    other = ds.copy(deep=deep)
    ds["v"].attrs["note"] = "later"
    ds["w"] = ("x", np.zeros(3))
    ds.attrs["title"] = "changed"
    del ds["s"]
    assert "note" not in other["v"].attrs
    assert "w" not in other
    assert "s" in other
    assert other.attrs["title"] == "t"
    # And the other way round.
    other["v"].attrs["mine"] = 1
    assert "mine" not in ds["v"].attrs


def test_view_survives_everything_done_to_its_source():
    ds = ar.Dataset({"v": ("x", np.arange(4.0))})
    child = ds["v"]
    held = child.values
    view = ds["v"].values
    ds["v"] = ("x", np.zeros(4))
    del child
    gc.collect()
    np.testing.assert_array_equal(held, np.arange(4.0))
    np.testing.assert_array_equal(view, np.arange(4.0))
    del ds
    gc.collect()
    np.testing.assert_array_equal(held, np.arange(4.0))
    assert not held.flags.writeable


def test_child_attribute_writes_still_reach_the_dataset():
    ds = sample()
    child = ds["v"]
    child.attrs["note"] = "n"
    assert ds["v"].attrs["note"] == "n"
    # Until the dataset replaces the variable.
    ds["v"] = ("x", np.zeros(3))
    child.attrs["late"] = "l"
    assert "late" not in ds["v"].attrs


def test_memory_does_not_grow_with_children():
    if sys.platform != "linux":
        pytest.skip("reads /proc")

    def rss() -> int:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * 4096

    ds = ar.Dataset({"v": (("x", "y"), np.ones((1000, 1000)))})
    before = rss()
    kept = [ds["v"] for _ in range(100)]
    grew = rss() - before
    del kept
    assert grew < 20_000_000  # 100 copies would be 800 MB


def test_lazy_variables_stay_lazy_when_taken(tmp_path):
    path = tmp_path / "lazy.nc"
    ar.Dataset({"v": ("x", np.arange(4.0))}).to_netcdf(path)
    ds = ar.open_dataset(path)
    child = ds["v"]
    assert ds.data_vars["v"].values.sum() == 6.0
    np.testing.assert_array_equal(child.values, np.arange(4.0))


# ----- item 2: `units` in attrs -----


def holders(ds: ar.Dataset):
    """A data variable and a coordinate, however they are reached."""
    yield "ds[k]", lambda: ds["v"]
    yield "ds.coords[k]", lambda: ds.coords["x"]
    yield "ds[coord]", lambda: ds["x"]


def test_attrs_show_the_unit():
    ds = sample()
    for label, get in holders(ds):
        a = get()
        assert a.attrs.get("units") == a.units, label
        assert "units" in a.attrs, label
        assert "units" in list(a.attrs), label
        assert dict(a.attrs)["units"] == a.units, label
    alone = ar.DataArray([1.0], dims="x", units="km")
    assert alone.attrs == {"units": "km"}
    assert ar.DataArray([1.0], dims="x").attrs == {}


def test_writing_units_sets_the_unit():
    ds = sample()
    ds["v"].attrs["units"] = "m"
    assert ds["v"].units == "m"
    assert ds["v"].attrs["units"] == "m"
    np.testing.assert_array_equal(ds["v"].values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(ds["v"].to("m").values, [0.0, 1.0, 2.0])  # no scaling

    ds.coords["x"].attrs["units"] = "km"
    assert ds["x"].units == "km"
    assert ds["x"].attrs["axis"] == "X"

    alone = ar.DataArray([1.0, 2.0], dims="x", units="km")
    alone.attrs["units"] = "m"
    assert alone.units == "m"
    alone.attrs.update(units="cm")
    assert alone.units == "cm"
    assert alone.assign_attrs(units="s").units == "s"
    assert alone.units == "cm"
    alone.attrs = {"units": "mm", "a": 1}
    assert alone.units == "mm" and alone.attrs["a"] == 1
    alone.attrs = {"a": 2}
    assert alone.units is None


def test_writing_unparsable_units_keeps_the_string():
    a = ar.DataArray([1.0], dims="x", units="km")
    with pytest.warns(UserWarning, match="kept as an attribute"):
        a.attrs["units"] = "no such unit"
    assert a.units is None
    assert a.attrs["units"] == "no such unit"


def test_removing_units_clears_the_unit():
    ds = sample()
    del ds["v"].attrs["units"]
    assert ds["v"].units is None
    assert "units" not in ds["v"].attrs
    a = ar.DataArray([1.0], dims="x", units="km", attrs={"b": 1})
    assert a.attrs.pop("units") == "km"
    assert a.units is None
    assert a.attrs == {"b": 1}
    a = ar.DataArray([1.0], dims="x", units="km", attrs={"b": 1})
    a.attrs.clear()
    assert a.units is None and a.attrs == {}


def test_attrs_rebuild_an_array_without_duplicating_the_unit():
    a = sample()["v"]
    b = ar.DataArray(a.values, dims=a.dims, attrs=dict(a.attrs))
    assert b.units == "km"
    assert b.identical(a.drop_vars("x").rename(None))
    ds = ar.Dataset({"v": (a.dims, a.values, dict(a.attrs))})
    assert ds["v"].units == "km"
    assert list(ds["v"].attrs) == ["units"]


def test_units_written_once(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    ds = sample()
    ds["v"].attrs["units"] = "m"
    path = tmp_path / "u.nc"
    ds[["v"]].to_netcdf(path)
    with netCDF4.Dataset(path) as f:
        assert f["v"].getncattr("units") == "m"
        assert f["v"].ncattrs().count("units") == 1
    back = ar.open_dataset(path)
    assert back["v"].units == "m"
    assert back["v"].attrs["units"] == "m"


def test_keep_attrs_false_keeps_the_unit():
    a = ar.DataArray([1.0, 2.0], dims="x", units="km", attrs={"b": 1})
    assert a.clip(ar.Quantity(0.0, "km"), keep_attrs=False).attrs == {"units": "km"}


def test_identical_sees_the_unit_once():
    a = ar.DataArray([1.0], dims="x", units="km")
    b = ar.DataArray([1.0], dims="x", attrs={"units": "km"})
    assert a.identical(b)
    b.attrs["units"] = "m"
    assert not a.identical(b)


def test_argsort_and_rank_results_have_no_unit():
    a = ar.DataArray([3.0, 1.0], dims="x", units="km")
    assert a.argsort().units is None
    assert a.rank("x").units is None


# ----- item 3: numpy arrays in arithmetic -----


def test_arithmetic_with_arrays():
    v = sample()["v"]
    for out in (v * np.ones(3), np.ones(3) * v):
        np.testing.assert_array_equal(out.values, [0.0, 1.0, 2.0])
        assert out.units == "km"
        assert out.dims == ("x",)
        assert out.name == "v"
        assert "x" in out.coords
    np.testing.assert_array_equal((v / np.array([1.0, 2.0, 4.0])).values, [0.0, 0.5, 0.5])
    assert (v / np.ones(3)).units == "km"
    np.testing.assert_array_equal(
        (np.array([1.0, 1.0, 1.0]) / (v + ar.Quantity(1, "km"))).values, [1, 0.5, 1 / 3]
    )
    assert (np.ones(3) / v).unit == ar.Unit("km-1")
    np.testing.assert_array_equal((v * [1.0, 2.0, 3.0]).values, [0.0, 2.0, 6.0])


def test_unit_checks_stay_with_arrays():
    v = sample()["v"]
    for f in (lambda: v + np.ones(3), lambda: np.ones(3) - v, lambda: v > np.ones(3)):
        with pytest.raises(ar.UnitError):
            f()
    plain = ar.DataArray([1.0, 2.0, 3.0], dims="x")
    np.testing.assert_array_equal((plain + np.ones(3)).values, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal((np.ones(3) - plain).values, [0.0, -1.0, -2.0])
    np.testing.assert_array_equal((plain > np.full(3, 2.0)).values, [False, False, True])
    np.testing.assert_array_equal((np.full(3, 2.0) > plain).values, [True, False, False])


def test_shape_mismatch_is_a_value_error():
    v = sample()["v"]
    for f in (lambda: v * np.ones(2), lambda: np.ones(2) * v, lambda: v * np.ones((2, 3))):
        with pytest.raises(ValueError, match="broadcast"):
            f()
    with pytest.raises(ValueError, match="broadcast"):
        sample()[["v"]] * np.ones(2)


def test_arrays_broadcast_by_position():
    a = ar.DataArray(np.ones((2, 3)), dims=("y", "x"))
    np.testing.assert_array_equal((a * np.arange(3.0)).values, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal((np.arange(3.0) * a).values, [[0, 1, 2], [0, 1, 2]])


def test_power_with_arrays():
    plain = ar.DataArray([1.0, 2.0, 3.0], dims="x", name="p", coords={"x": [0, 1, 2]})
    out = plain ** np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out.values, [1.0, 4.0, 27.0])
    assert out.name == "p" and "x" in out.coords
    np.testing.assert_array_equal((np.array([2.0, 2.0, 2.0]) ** plain).values, [2.0, 4.0, 8.0])
    v = sample()["v"]
    squared = v ** np.full(3, 2.0)
    np.testing.assert_array_equal(squared.values, [0.0, 1.0, 4.0])
    assert squared.unit == ar.Unit("km2")
    with pytest.raises(ar.UnitError):
        v ** np.array([1.0, 2.0, 3.0])


def test_in_place_through_a_dataset():
    ds = sample()
    ds["v"] *= np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ds["v"].values, [0.0, 2.0, 6.0])
    assert ds["v"].units == "km"
    with pytest.raises(ar.UnitError):
        ds["v"] += np.ones(3)


def test_datasets_with_arrays():
    ds = ar.Dataset(
        {"a": ("x", np.arange(3.0), {"units": "km"}), "b": ("x", np.ones(3))},
    )
    for out in (ds * np.full(3, 2.0), np.full(3, 2.0) * ds):
        np.testing.assert_array_equal(out["a"].values, [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(out["b"].values, [2.0, 2.0, 2.0])
        assert out["a"].units == "km"
    with pytest.raises(ar.UnitError):
        ds + np.ones(3)
    np.testing.assert_array_equal(
        (ds[["b"]] > np.array([0.0, 2.0, 0.0]))["b"].values, [True, False, True]
    )
    np.testing.assert_array_equal((ds[["b"]] ** np.full(3, 3.0))["b"].values, [1.0, 1.0, 1.0])


def test_float32_with_an_array_promotes():
    a = ar.DataArray(np.ones(2, np.float32), dims="x")
    assert (a * np.ones(2)).dtype == np.float64
    assert (a * np.ones(2, np.float32)).dtype == np.float32


# ----- item 4: variables named after their dimension -----


def check_index(ds: ar.Dataset) -> None:
    assert "w" in ds.coords
    assert "w" not in ds.data_vars
    np.testing.assert_array_equal(ds["q"].sel(w=1.0).values, 1.0)


def test_dimension_named_variable_is_a_coordinate():
    a = np.arange(2.0)
    check_index(ar.Dataset({"w": ("w", a), "q": ("w", a)}))
    u = ar.Dataset({"q": ("w", a)})
    u.update({"w": ("w", a)})
    check_index(u)
    s = ar.Dataset({"q": ("w", a)})
    s["w"] = ("w", a)
    check_index(s)
    check_index(ar.Dataset({"q": ("w", a)}).assign(w=("w", a)))
    check_index(ar.merge([ar.Dataset({"q": ("w", a)}), ar.Dataset({"w": ("w", a)})]))
    check_index(ar.Dataset({"w": ar.DataArray(a, dims="w"), "q": ("w", a)}))


def test_data_variable_replaced_by_an_index():
    ds = ar.Dataset({"w": ("x", np.arange(2.0)), "q": ("x", np.arange(2.0))})
    assert "w" in ds.data_vars
    with pytest.raises(ValueError):
        ds["w"] = ("x", np.arange(3.0))
    assert ds["w"].sizes == {"x": 2}
    ds = ar.Dataset({"w": ("x", np.arange(2.0))})
    ds["w"] = ("w", np.arange(5.0))
    assert "w" in ds.coords and "w" not in ds.data_vars


def test_other_shapes_are_data_variables():
    ds = ar.Dataset({"w": (("w", "x"), np.ones((1, 2))), "v": ("x", np.ones(2))})
    assert "w" in ds.data_vars


# ----- item 5: assign_coords replaces -----


def test_assign_coords_replaces():
    d = sample()
    out = d.assign_coords(x=("x", np.arange(3.0) * 2))
    np.testing.assert_array_equal(out.coords["x"].values, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(d.coords["x"].values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(out["v"].coords["x"].values, [0.0, 2.0, 4.0])
    assert float(out["v"].sel(x=2.0).values) == 1.0
    a = d["v"].assign_coords(x=("x", np.arange(3.0) * 2))
    np.testing.assert_array_equal(a.coords["x"].values, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(d["v"].coords["x"].values, [0.0, 1.0, 2.0])


def test_assign_coords_refuses_another_length():
    d = sample()
    with pytest.raises(ValueError):
        d.assign_coords(x=("x", np.arange(4.0)))
    with pytest.raises(ValueError):
        d["v"].assign_coords(x=("x", np.arange(4.0)))
    assert d.sizes == {"x": 3}


def test_assign_coords_takes_over_a_data_variable():
    out = ar.Dataset({"a": ("x", np.arange(2.0)), "b": ("x", np.arange(2.0))}).assign_coords(
        a=("x", np.ones(2))
    )
    assert "a" in out.coords and "a" not in out.data_vars


# ----- item 6: boolean masks -----


def test_boolean_masks_select():
    d = sample()
    mask = d["s"] == "a"
    v = d["v"]
    for out in (v.sel(x=mask), v.isel(x=mask), d.sel(x=mask)["v"], d.isel(x=mask)["v"]):
        np.testing.assert_array_equal(out.values, [0.0, 2.0])
        np.testing.assert_array_equal(out.coords["x"].values, [0.0, 2.0])
        assert out.units == "km"
    np.testing.assert_array_equal(v.sel(x=mask.values).values, [0.0, 2.0])
    np.testing.assert_array_equal(v.isel(x=mask.values).values, [0.0, 2.0])


def test_unlabelled_mask_selects():
    v = sample()["v"]
    np.testing.assert_array_equal(
        v.isel(x=ar.DataArray([True, False, False], dims="x")).values, [0.0]
    )


def test_mask_with_another_coordinate_raises():
    d = sample()
    wrong = ar.DataArray([True, False, True], dims="x", coords={"x": [5.0, 6.0, 7.0]})
    for f in (d["v"].sel, d["v"].isel, d.sel, d.isel):
        with pytest.raises(ValueError, match="conflicts"):
            f(x=wrong)


@pytest.mark.parametrize(
    "bad",
    [
        ar.DataArray([True, False], dims="x"),
        ar.DataArray([True, False, True], dims="y"),
        ar.DataArray(np.ones((3, 2), bool), dims=("x", "y")),
    ],
)
def test_misshapen_mask_raises(bad):
    with pytest.raises(ValueError, match="boolean indexer"):
        sample()["v"].isel(x=bad)


@pytest.mark.parametrize("method", ["sel", "isel"])
def test_plain_mask_of_another_length_raises(method):
    with pytest.raises(ValueError, match="boolean indexer"):
        getattr(sample()["v"], method)(x=np.array([True, False]))
    with pytest.raises(ValueError, match="boolean indexer"):
        getattr(sample(), method)(x=np.array([True, False, True, True]))


def test_where_with_an_array():
    d = sample()
    v = d["v"]
    out = v.where(v.values > 0)
    assert np.isnan(out.values[0])
    np.testing.assert_array_equal(out.values[1:], [1.0, 2.0])
    assert out.units == "km"
    with pytest.raises(ValueError, match="broadcast"):
        v.where(np.array([True, False]))
    ds = d[["v"]].where(np.array([False, True, True]))
    assert np.isnan(ds["v"].values[0])
    assert ds["v"].values[1] == 1.0


@pytest.mark.parametrize("as_array", [False, True])
def test_where_drop(as_array):
    d = sample()
    v = d["v"]
    mask = v > ar.Quantity(0.5, "km")
    cond = mask.values if as_array else mask
    out = v.where(cond, drop=True)
    np.testing.assert_array_equal(out.values, [1.0, 2.0])
    if not as_array:
        np.testing.assert_array_equal(out.coords["x"].values, [1.0, 2.0])
        np.testing.assert_array_equal(d[["v"]].where(cond, drop=True)["v"].values, [1.0, 2.0])
    with pytest.raises(ValueError, match="other"):
        v.where(cond, ar.Quantity(0, "km"), drop=True)


def test_where_drop_2d_keeps_partial_rows():
    a = ar.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"))
    out = a.where(a.values > 3, drop=True)
    assert out.shape == (1, 2)
    np.testing.assert_array_equal(out.values, [[4.0, 5.0]])


# ----- item 7: unit parsing and registration -----


@pytest.mark.parametrize(
    ("braced", "plain"),
    [
        ("cm^{-3}", "cm^-3"),
        ("s^{-1}", "s-1"),
        ("eV cm^{-3} s^{-1}", "eV cm-3 s-1"),
        ("m^{2}", "m2"),
    ],
)
def test_braced_exponents(braced, plain):
    assert ar.Unit(braced) == ar.Unit(plain)
    assert str(ar.Unit(braced)) == braced


@pytest.mark.parametrize("bad", ["m^{2", "m^{}", "m^{x}", "m{2}", "{2}"])
def test_bad_braces(bad):
    with pytest.raises(ar.UnitError):
        ar.Unit(bad)


def test_photons_count():
    assert ar.Unit("photon") == ar.Unit("1") == ar.Unit("photons")
    assert ar.Quantity(1, "photons cm-2 s-1").to("m-2 s-1").magnitude == 1e4
    # `ph` stays the phot, 10⁴ lx: not a photon.
    assert ar.Quantity(1, "ph").to("lx").magnitude == 1e4


def test_define_unit():
    unit = ar.units.define_unit("PyTestTECU", "1e16 m^-2")
    assert str(unit) == "PyTestTECU"
    assert ar.Unit("PyTestTECU") == ar.Unit("1e16 m-2")
    assert ar.Quantity(1, "PyTestTECU").to("m-2").magnitude == 1e16
    assert ar.Quantity(1, "kPyTestTECU").to("PyTestTECU").magnitude == 1000
    assert ar.Quantity(1e16, "m-2").to("PyTestTECU").magnitude == 1
    # The same definition again is fine.
    ar.units.define_unit("PyTestTECU", ar.Unit("1e16 m-2"))
    assert ar.define_unit is ar.units.define_unit


def test_define_unit_conflicts():
    ar.units.define_unit("PyTestConflict", "1 m")
    for existing, definition in [("PyTestConflict", "2 m"), ("km", "3 m"), ("Pa", "1 s")]:
        with pytest.raises(ar.UnitError, match="already exists"):
            ar.units.define_unit(existing, definition)
    ar.units.define_unit("km", "1000 m")  # what it already is
    with pytest.raises(ar.UnitError):
        ar.units.define_unit("m2", "1 s")  # would read back as m squared
    with pytest.raises(ar.UnitError):
        ar.units.define_unit("PyTestBad", "no such unit")
    with pytest.raises(ar.UnitError):
        ar.Unit("PyTestBad")


def test_custom_unit_round_trips(tmp_path):
    ar.units.define_unit("PyTestRoundTrip", "1e16 m^-2")
    path = tmp_path / "custom.nc"
    ds = ar.Dataset({"tec": ("x", np.arange(2.0), {"units": "PyTestRoundTrip"})})
    ds.to_netcdf(path)
    back = ar.open_dataset(path)
    assert back["tec"].units == "PyTestRoundTrip"
    np.testing.assert_array_equal(back["tec"].to("m-2").values, [0.0, 1e16])


def test_custom_unit_to_astropy():
    pytest.importorskip("astropy")
    ar.units.define_unit("PyTestAstropy", "1e16 m^-2")
    a = ar.to_astropy(ar.Unit("PyTestAstropy"))
    assert ar.from_astropy(a) == ar.Unit("PyTestAstropy")


def test_a_redefined_failure_is_not_remembered():
    with pytest.raises(ar.UnitError):
        ar.Unit("PyTestLater")
    ar.units.define_unit("PyTestLater", "1 m")
    assert ar.Unit("PyTestLater") == ar.Unit("m")


# ----- item 8: interp at a scalar -----


def test_interp_scalar():
    d = sample()
    out = d["v"].interp(x=1.5)
    assert out.ndim == 0
    assert float(out.values) == 1.5
    assert out.units == "km"
    assert float(out.coords["x"].values) == 1.5
    ds = d[["v"]].interp(x=1.5)
    assert ds["v"].ndim == 0
    assert float(ds["x"].values) == 1.5
    np.testing.assert_array_equal(d["v"].interp(x=[1.5]).values, [1.5])
    with_quantity = d["v"].interp(x=ar.Quantity(1.5, "m"))
    assert float(with_quantity.values) == 1.5


# ----- item 9: None attributes -----


@pytest.mark.parametrize(
    "make",
    [
        lambda: ar.Dataset(attrs={"xuvfac": None}),
        lambda: ar.DataArray([1.0], dims="x", attrs={"xuvfac": None}),
        lambda: ar.Dataset({"v": ("x", [1.0], {"xuvfac": None})}),
        lambda: ar.DataArray([1.0], dims="x").assign_attrs(xuvfac=None),
    ],
)
def test_none_attribute_is_refused_by_name(make):
    with pytest.raises(ValueError, match="xuvfac.*None"):
        make()


def test_setting_none_attribute_names_it():
    a = ar.DataArray([1.0], dims="x")
    with pytest.raises(ValueError, match="xuvfac"):
        a.attrs["xuvfac"] = None
    d = ar.Dataset()
    with pytest.raises(ValueError, match="xuvfac"):
        d.attrs["xuvfac"] = None


# ----- item 10: clip with a bare number -----


def test_clip_with_a_bare_number_is_in_the_data_unit():
    v = sample()["v"]
    np.testing.assert_array_equal(v.clip(min=0.5).values, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(v.clip(max=1.5).values, [0.0, 1.0, 1.5])
    assert v.clip(min=0.5).units == "km"
    ds = sample()[["v"]].clip(min=0.5)
    np.testing.assert_array_equal(ds["v"].values, [0.5, 1.0, 2.0])
    t = ar.DataArray([0.0, 20.0], dims="x", units="degC")
    np.testing.assert_array_equal(t.clip(min=5.0).values, [5.0, 20.0])
    # Comparisons stay strict.
    with pytest.raises(ar.UnitError):
        _ = v > 0.5


# ----- item 11: copy=False -----


def test_copy_false_shares_memory():
    source = np.arange(6.0).reshape(2, 3)
    ds = ar.Dataset({"v": (("y", "x"), source)}, coords={"x": ("x", source[0])}, copy=False)
    assert np.shares_memory(ds["v"].values, source)
    assert np.shares_memory(ds["x"].values, source)
    assert not ds["v"].values.flags.writeable
    assert source.flags.writeable  # still the caller's
    da = ar.DataArray(source, dims=("y", "x"), copy=False)
    assert np.shares_memory(da.values, source)
    assert not np.shares_memory(ar.DataArray(source, dims=("y", "x")).values, source)
    assert not np.shares_memory(ar.Dataset({"v": (("y", "x"), source)})["v"].values, source)


def test_copy_false_keeps_the_layout_and_computes():
    f = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))
    ds = ar.Dataset({"v": (("a", "b", "c"), f)}, copy=False)
    values = ds["v"].values
    assert values.flags.f_contiguous and not values.flags.c_contiguous
    assert np.shares_memory(values, f)
    np.testing.assert_array_equal(values, f)
    np.testing.assert_array_equal((ds["v"] * 2).values, f * 2)
    np.testing.assert_array_equal(ds["v"].isel(a=1).values, f[1])
    np.testing.assert_array_equal(ds["v"].sum("b").values, f.sum(1))
    np.testing.assert_array_equal(ds["v"].transpose("c", "a", "b").values, f.transpose(2, 0, 1))
    np.testing.assert_array_equal(ds["v"].stack(s=("a", "b")).values.shape, (4, 6))


def test_copy_false_round_trips_fortran_order(tmp_path):
    f = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    path = tmp_path / "f.nc"
    ar.Dataset({"v": (("a", "b"), f)}, copy=False).to_netcdf(path)
    np.testing.assert_array_equal(ar.open_dataset(path)["v"].values, f)


def test_copy_false_keeps_strided_and_reversed_views():
    base = np.arange(12.0)
    view = base[::-2]
    da = ar.DataArray(view, dims="x", copy=False)
    assert np.shares_memory(da.values, base)
    np.testing.assert_array_equal(da.values, view)
    np.testing.assert_array_equal(da.isel(x=slice(1, 3)).values, view[1:3])
    np.testing.assert_array_equal((da + 1).values, view + 1)


def test_copy_false_keeps_the_source_alive():
    source = np.arange(4.0)
    ref = weakref.ref(source)
    ds = ar.Dataset({"v": ("x", source)}, copy=False)
    del source
    gc.collect()
    assert ref() is not None
    np.testing.assert_array_equal(ds["v"].values, np.arange(4.0))
    del ds
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize(
    "values",
    [
        np.arange(3, dtype=np.float32),
        np.arange(3, dtype=np.int16),
        np.array([True, False]),
        np.array([1 + 2j, 3j]),
        np.array(["2000-01-01", "NaT"], "datetime64[ns]"),
        np.array([1, 2], "timedelta64[ns]"),
    ],
)
def test_copy_false_element_types(values):
    da = ar.DataArray(values, dims="x", copy=False)
    assert np.shares_memory(da.values, values)
    np.testing.assert_array_equal(da.values, values)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(3, dtype=">f8"),
        np.arange(3, dtype=np.float16),
        np.array(["a", "b"]),
        [1.0, 2.0],
        np.array(["2000-01-01"], "datetime64[s]"),
    ],
)
def test_copy_false_copies_what_it_must(values):
    da = ar.DataArray(values, dims="x", copy=False)
    assert da.shape == (len(values),)
    np.testing.assert_array_equal(da.values, ar.DataArray(values, dims="x").values)


def test_writing_in_rust_style_put_leaves_the_source():
    source = np.arange(3.0)
    da = ar.DataArray(source, dims="x", copy=False)
    out = da.put(9.0, x=0)
    assert out.values[0] == 9.0
    assert source[0] == 0.0
    assert da.copy(deep=True).values[0] == 0.0
    assert not np.shares_memory(da.copy(deep=True).values, source)


# ----- item 12: matplotlib is optional -----


def run_without_matplotlib(code: str) -> subprocess.CompletedProcess[str]:
    block = """
        import importlib.util
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "matplotlib" or name.startswith("matplotlib."):
                    raise ImportError(name)
        sys.meta_path.insert(0, Block())
        found = importlib.util.find_spec
        importlib.util.find_spec = lambda n, p=None: None if n == "matplotlib" else found(n, p)
        """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(block) + textwrap.dedent(code)],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


def test_import_works_without_matplotlib():
    done = run_without_matplotlib(
        """
        import sys
        import annoray as ar
        (ar.DataArray([1.0], dims="x") * 2).sum()
        assert not [m for m in sys.modules if m.split(".")[0] == "matplotlib"]
        """
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    "access", ['ar.DataArray([1.0], dims="x").plot', 'ar.Dataset({"v": ("x", [1.0])}).plot']
)
def test_plot_without_matplotlib_names_the_extra(access):
    done = run_without_matplotlib(f"import annoray as ar\n{access}\n")
    assert done.returncode != 0
    assert "ImportError" in done.stderr
    assert "annoray[plot]" in done.stderr


# ----- selections and transposes are views; strided values write correctly -----


def cube() -> ar.Dataset:
    values = np.arange(120.0).reshape(2, 3, 4, 5)
    return ar.Dataset(
        {"v": (("a", "b", "c", "d"), values, {"units": "km"})},
        coords={"a": ("a", [10.0, 20.0]), "d": ("d", list("vwxyz"))},
    )


@pytest.mark.parametrize(
    "select",
    [
        lambda o: o.isel(d=0),
        lambda o: o.isel(a=1, c=slice(1, 3)),
        lambda o: o.isel(b=slice(None, None, -1)),
        lambda o: o.isel(c=slice(None, None, 2)).isel(d=1),
        lambda o: o.sel(a=20.0),
        lambda o: o.sel(d="x"),
        lambda o: o.transpose("d", "c", "b", "a"),
        lambda o: o.isel(d=0).transpose("c", "b", "a"),
        lambda o: o.transpose("d", "c", "b", "a").isel(a=1),
    ],
)
def test_selections_are_views(select):
    ds = cube()
    v = ds["v"]
    numpy_view = v.values
    got = select(v)
    assert np.shares_memory(got.values, numpy_view)
    assert not got.values.flags.writeable
    assert got.units == "km"
    assert np.shares_memory(select(ds)["v"].values, numpy_view)
    # The values are what a copy would hold, and what the source still holds.
    np.testing.assert_array_equal(numpy_view, np.arange(120.0).reshape(2, 3, 4, 5))
    copied = ar.DataArray(numpy_view.copy(), dims=v.dims, coords=dict(v.coords), units="km")
    np.testing.assert_array_equal(got.values, select(copied).values)
    # Everything keeps working on the view, equal to the same on a fresh contiguous array.
    fresh = ar.DataArray(np.array(got.values, order="C"), dims=got.dims, units="km")
    np.testing.assert_array_equal((got * 2).values, (fresh * 2).values)
    np.testing.assert_array_equal(got.sum().values, fresh.sum().values)
    np.testing.assert_array_equal(got.mean(got.dims[0]).values, fresh.mean(got.dims[0]).values)


def test_view_outlives_its_source_and_replacement():
    ds = cube()
    picked = ds["v"].isel(d=0)
    held = picked.values
    ds["v"] = (("a", "b", "c", "d"), np.zeros((2, 3, 4, 5)))
    del ds
    gc.collect()
    np.testing.assert_array_equal(held, np.arange(120.0).reshape(2, 3, 4, 5)[..., 0])


def test_deep_copy_of_a_selection_holds_only_the_selection():
    source = np.arange(120.0).reshape(2, 3, 4, 5)
    ref = weakref.ref(source)
    picked = ar.DataArray(source, dims=("a", "b", "c", "d"), copy=False).isel(d=0)
    assert ref() is not None
    deep = picked.copy(deep=True)
    del picked, source
    gc.collect()
    assert ref() is None  # nothing holds the whole array any more
    assert deep.values.size == 24
    np.testing.assert_array_equal(deep.values, np.arange(120.0).reshape(2, 3, 4, 5)[..., 0])


def test_put_on_a_view_leaves_the_source():
    v = cube()["v"]
    before = v.values.copy()
    out = v.isel(d=0).put(ar.Quantity(-1.0, "km"), a=0, b=0, c=0)
    assert out.values[0, 0, 0] == -1.0
    np.testing.assert_array_equal(v.values, before)


def test_selections_of_copy_false_arrays_are_views_of_the_source():
    source = np.arange(24.0).reshape(2, 3, 4)
    da = ar.DataArray(source, dims=("a", "b", "c"), copy=False)
    assert np.shares_memory(da.isel(c=1).values, source)
    assert np.shares_memory(da.transpose("c", "b", "a").values, source)
    np.testing.assert_array_equal(da.isel(c=1).values, source[..., 1])
    np.testing.assert_array_equal(da.transpose("c", "b", "a").values, source.T)


def strided_cases():
    base = np.arange(2.0 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    yield "moveaxis", np.moveaxis(base, 0, -1)
    yield "transposed", base.T
    yield "stepped", base[:, ::2, :, 1:4]
    yield "reversed", base[::-1, :, ::-1, :]
    big = np.arange(50.0 * 60 * 70).reshape(50, 60, 70)
    yield "beyond a copy block", np.moveaxis(big, 0, -1)


@pytest.mark.parametrize(("label", "values"), list(strided_cases()), ids=lambda x: str(x)[:20])
@pytest.mark.parametrize(
    ("write", "read"),
    [
        (lambda ds, p: ds.to_netcdf(p), ar.open_dataset),
        (lambda ds, p: ds.to_netcdf(p, format="NETCDF3_64BIT"), ar.open_dataset),
        (lambda ds, p: ds.to_netcdf(p, encoding={"v": {"zlib": True}}), ar.open_dataset),
        (lambda ds, p: ds.to_zarr(p), ar.open_zarr),
    ],
    ids=["netcdf4", "netcdf3", "chunked", "zarr"],
)
def test_strided_values_write_in_logical_order(tmp_path, label, values, write, read):
    dims = [f"d{i}" for i in range(values.ndim)]
    ds = ar.Dataset({"v": (dims, values)}, copy=False)
    path = tmp_path / "s"
    write(ds, path)
    np.testing.assert_array_equal(read(path)["v"].values, values)


def test_transposed_views_write_in_their_order(tmp_path):
    v = cube()["v"]
    path = tmp_path / "t.nc"
    t = v.transpose("d", "c", "b", "a")
    ar.Dataset({"t": t}).to_netcdf(path)
    back = ar.open_dataset(path)["t"]
    assert back.dims == t.dims
    assert back.units == "km"
    np.testing.assert_array_equal(back.values, v.values.T)


# ----- update(copy=False) -----


def test_update_without_a_copy():
    source = np.arange(3.0)
    ds = ar.Dataset(coords={"x": ("x", np.arange(3.0))})
    assert ds.update({"v": ("x", source, {"units": "km"})}, copy=False) is ds
    assert np.shares_memory(ds["v"].values, source)
    assert ds["v"].units == "km"
    ds.update({"w": ("x", source)})
    assert not np.shares_memory(ds["w"].values, source)
    ds.update({"x": ("x", source)}, copy=False)
    assert np.shares_memory(ds["x"].values, source)


# ----- TECU is built in -----


def test_tecu_is_built_in():
    assert ar.Unit("TECU") == ar.Unit("1e16 m-2")
    assert str(ar.Unit("TECU")) == "TECU"
    assert ar.Quantity(1, "TECU").to("m-2").magnitude == 1e16
    assert ar.Quantity(1, "mTECU").to("TECU").magnitude == 1e-3
    ar.define_unit("TECU", "1e16 m^-2")  # still fine for code that defines it
    with pytest.raises(ar.UnitError, match="already exists"):
        ar.define_unit("TECU", "1e15 m^-2")


# ----- reductions walk memory, whatever the layout -----


def reduction_cases():
    values = np.random.default_rng(1).random((6, 7, 8, 5))
    values[1, 2, 3, 4] = np.nan
    values[4, :, 2, 1] = np.nan
    source = ar.DataArray(values, dims=("a", "b", "c", "d"), units="km")
    yield "transposed", source.transpose("d", "c", "b", "a")
    yield "rotated", source.transpose("c", "a", "d", "b")
    yield "reversed", source.isel(a=slice(None, None, -1), c=slice(None, None, -1))
    yield "stepped", source.isel(b=slice(None, None, 2), d=slice(1, 4))
    yield "selected", source.isel(d=2)
    yield "both", source.isel(a=slice(None, None, -1)).transpose("d", "b", "c", "a")
    tall = np.random.default_rng(2).random((40, 50, 30))
    # Enough kept values for the row kernels, in each order.
    yield "large", ar.DataArray(tall, dims=("a", "b", "c")).transpose("c", "b", "a")
    yield "large reversed", ar.DataArray(tall, dims=("a", "b", "c")).isel(a=slice(None, None, -1))


@pytest.mark.parametrize(("label", "view"), list(reduction_cases()), ids=lambda x: str(x)[:16])
@pytest.mark.parametrize("op", ["sum", "mean", "std", "var", "min", "max", "median", "count"])
def test_reductions_of_views_equal_those_of_contiguous_values(label, view, op):
    fresh = ar.DataArray(np.array(view.values, order="C"), dims=view.dims, units=view.units)
    dims = list(view.dims)
    choices = [None, *dims, tuple(dims[:2]), tuple(dims[-2:]), (dims[-1], dims[0]), tuple(dims)]
    if len(dims) > 2:
        choices.append((dims[1], dims[-1]))
    for dim in choices:
        got = getattr(view, op)(dim)
        want = getattr(fresh, op)(dim)
        assert got.dims == want.dims, dim
        assert got.units == want.units, dim
        np.testing.assert_allclose(got.values, want.values, rtol=1e-12, err_msg=str(dim))
        assert got.values.shape == want.values.shape


@pytest.mark.parametrize(("label", "view"), list(reduction_cases()), ids=lambda x: str(x)[:16])
def test_reduction_options_on_views(label, view):
    fresh = ar.DataArray(np.array(view.values, order="C"), dims=view.dims)
    plain = ar.DataArray(view.values, dims=view.dims, copy=False)
    d = view.dims[0]
    for got, want in [
        (plain.sum(d, skipna=False), fresh.sum(d, skipna=False)),
        (plain.sum(d, min_count=view.sizes[d]), fresh.sum(d, min_count=view.sizes[d])),
        (plain.std(d, ddof=1), fresh.std(d, ddof=1)),
        (plain.prod(d), fresh.prod(d)),
    ]:
        np.testing.assert_allclose(got.values, want.values, rtol=1e-12)


def test_integer_reductions_of_views_are_exact():
    values = np.arange(120, dtype=np.int64).reshape(2, 3, 4, 5) ** 2
    source = ar.DataArray(values, dims=("a", "b", "c", "d"))
    t = source.transpose("d", "c", "b", "a")
    assert int(t.sum().values) == int(values.sum())
    np.testing.assert_array_equal(t.sum("a").values, values.T.sum(3))
    np.testing.assert_array_equal(t.max("b").values, values.T.max(2))
    r = source.isel(a=slice(None, None, -1))
    np.testing.assert_array_equal(r.sum("b").values, values[::-1].sum(1))


def test_boolean_reductions_of_views():
    values = np.random.default_rng(3).random((5, 6, 7)) > 0.9
    t = ar.DataArray(values, dims=("a", "b", "c")).transpose("c", "a", "b")
    np.testing.assert_array_equal(t.any("a").values, values.transpose(2, 0, 1).any(1))
    np.testing.assert_array_equal(t.all("c").values, values.transpose(2, 0, 1).all(0))


@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_arg_reductions_answer_in_logical_positions(op):
    # Ties make the answer depend on which comes first, which is a logical question.
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    t = ar.DataArray(values, dims=("a", "b")).transpose("b", "a")
    r = ar.DataArray(values, dims=("a", "b")).isel(b=slice(None, None, -1))
    np.testing.assert_array_equal(getattr(t, op)("a").values, getattr(values.T, op)(1))
    np.testing.assert_array_equal(getattr(t, op)("b").values, getattr(values.T, op)(0))
    np.testing.assert_array_equal(getattr(r, op)("b").values, getattr(values[:, ::-1], op)(1))


def test_reduction_results_keep_working():
    v = ar.DataArray(np.arange(24.0).reshape(2, 3, 4), dims=("a", "b", "c"), name="n", units="m")
    out = v.transpose("c", "b", "a").sum("b")
    assert out.dims == ("c", "a")
    assert out.units == "m"
    np.testing.assert_array_equal((out * 2).values, np.arange(24.0).reshape(2, 3, 4).sum(1).T * 2)
    np.testing.assert_array_equal(out.isel(a=0).values, np.arange(24.0).reshape(2, 3, 4).sum(1)[0])


@pytest.mark.parametrize("dates", ["datetime64[ns]", "timedelta64[ns]"])
def test_time_reductions_of_views(dates):
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 4).astype(dates)
    t = ar.DataArray(values, dims=("a", "b", "c")).transpose("c", "b", "a")
    fresh = ar.DataArray(values.T.copy(), dims=("c", "b", "a"))
    np.testing.assert_array_equal(t.max("a").values, values.T.max(2))
    np.testing.assert_array_equal(t.mean("c").values, fresh.mean("c").values)


NO_COPY = """
import resource
import numpy as np
import annoray as ar

values = np.ones((200, 250, 500), np.float64)
da = ar.DataArray(values, dims=("a", "b", "c"), copy=False).transpose("c", "b", "a")
reversed_ = ar.DataArray(values, dims=("a", "b", "c"), copy=False).isel(b=slice(None, None, -1))
stepped = ar.DataArray(values, dims=("a", "b", "c"), copy=False).isel(c=slice(None, None, 2))
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
base = peak()
da.sum()
da.mean()
da.sum("a")
da.max("b")
reversed_.mean()
stepped.mean()
print((peak() - base) * 1024 / values.nbytes)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss units differ by platform")
def test_reductions_of_views_do_not_copy_the_values():
    done = subprocess.run(
        [sys.executable, "-c", NO_COPY], capture_output=True, encoding="utf-8", check=True
    )
    assert float(done.stdout.strip()) < 0.25


# ----- the slice and row kernels agree with numpy, wherever the awkward values fall -----

# Around a kernel's lanes, its shortest run and a thread's block.
KERNEL_LENGTHS = [1, 2, 15, 16, 17, 63, 64, 65, 79, 80, 1000, 65535, 65536, 65537, 200_001]


def awkward(n: int, dtype: Any, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(dtype)
    spots = {"first": [0], "last": [n - 1], "middle": [n // 2]}
    if kind in spots:
        x[spots[kind]] = np.nan
    elif kind == "many":
        x[rng.random(n) < 0.3] = np.nan
    elif kind == "all":
        x[:] = np.nan
    elif kind == "inf":
        x[n // 2] = np.inf
        x[0] = -np.inf
    elif kind == "only inf":
        x[:] = np.inf
    return x


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", KERNEL_LENGTHS)
@pytest.mark.parametrize(
    "kind", ["none", "first", "last", "middle", "many", "all", "inf", "only inf"]
)
def test_whole_array_kernels_agree_with_numpy(dtype, n, kind):
    x = awkward(n, dtype, kind, n)
    da = ar.DataArray(x, dims="x")
    has_valid = not np.isnan(x).all()
    wide = x.astype(np.float64)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cases = {
            "min": (da.min(), np.nanmin(x) if has_valid else np.nan),
            "max": (da.max(), np.nanmax(x) if has_valid else np.nan),
            "sum": (da.sum(), np.nansum(wide)),
            "mean": (da.mean(), np.nanmean(wide)),
            "var": (da.var(), np.nanvar(wide)),
            "std": (da.std(), np.nanstd(wide)),
            "min keep": (da.min(skipna=False), x.min()),
            "max keep": (da.max(skipna=False), x.max()),
            "sum keep": (da.sum(skipna=False), wide.sum()),
            "mean keep": (da.mean(skipna=False), wide.mean()),
        }
    for label, (got, want) in cases.items():
        rtol = 1e-5 if dtype is np.float32 else 1e-9
        if np.isinf(want) or np.isnan(want):
            np.testing.assert_equal(float(got.values), float(want), err_msg=label)
        else:
            np.testing.assert_allclose(float(got.values), want, rtol=rtol, atol=1e-5, err_msg=label)
        assert got.dtype == x.dtype, label


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [2, 16, 64, 80, 1000, 70_000])
def test_extremes_tell_signed_zeros_apart(dtype, n):
    zeros = np.zeros(n, dtype)
    assert not np.signbit(ar.DataArray(zeros, dims="x").min().values)
    assert not np.signbit(ar.DataArray(zeros, dims="x").max().values)
    assert np.signbit(ar.DataArray(-zeros, dims="x").min().values)
    assert np.signbit(ar.DataArray(-zeros, dims="x").max().values)
    mixed = zeros.copy()
    mixed[n // 2] = -0.0
    assert np.signbit(ar.DataArray(mixed, dims="x").min().values)
    assert not np.signbit(ar.DataArray(mixed, dims="x").max().values)
    mixed = -zeros
    mixed[n - 1] = 0.0
    assert np.signbit(ar.DataArray(mixed, dims="x").min().values)
    assert not np.signbit(ar.DataArray(mixed, dims="x").max().values)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
@pytest.mark.parametrize("n", KERNEL_LENGTHS)
def test_integer_extremes_agree_with_numpy(dtype, n):
    info = np.iinfo(dtype)
    x = np.random.default_rng(n).integers(info.min, info.max, n, dtype=dtype, endpoint=True)
    da = ar.DataArray(x, dims="x")
    assert da.min().values == x.min() and da.min().dtype == x.dtype
    assert da.max().values == x.max() and da.max().dtype == x.dtype
    # At each end of the range, in the last place looked at.
    x[n - 1] = info.max
    assert ar.DataArray(x, dims="x").max().values == info.max
    x[n - 1] = info.min
    assert ar.DataArray(x, dims="x").min().values == info.min


def test_sum_of_negative_zeros_and_of_nothing():
    n = 1000
    assert not np.signbit(ar.DataArray(np.zeros(n) * -1.0, dims="x").sum().values)
    assert ar.DataArray(np.full(n, np.nan), dims="x").sum().values == 0.0
    assert np.isnan(ar.DataArray(np.full(n, np.nan), dims="x").mean().values)
    assert np.isnan(ar.DataArray(np.full(n, np.nan), dims="x").sum(min_count=1).values)
    assert ar.DataArray(np.zeros(0), dims="x").sum().values == 0.0
    assert np.isnan(ar.DataArray(np.zeros(0), dims="x").mean().values)
    with pytest.raises(ValueError):
        ar.DataArray(np.zeros(0), dims="x").max()


@pytest.mark.parametrize("nans", ["none", "one row", "scattered", "whole column", "all"])
@pytest.mark.parametrize("shape", [(30, 300), (300, 30, 2), (5, 40_000)])
def test_row_kernels_count_what_they_skip(nans, shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape)
    if nans == "one row":
        x[1] = np.nan
    elif nans == "scattered":
        x[rng.random(shape) < 0.01] = np.nan
    elif nans == "whole column":
        x[(slice(None), 3)] = np.nan
    elif nans == "all":
        x[:] = np.nan
    dims = tuple(f"d{i}" for i in range(len(shape)))
    da = ar.DataArray(x, dims=dims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for axis, dim in enumerate(dims):
            np.testing.assert_allclose(da.sum(dim).values, np.nansum(x, axis), rtol=1e-11)
            np.testing.assert_allclose(da.mean(dim).values, np.nanmean(x, axis), rtol=1e-11)
            np.testing.assert_allclose(da.std(dim).values, np.nanstd(x, axis), rtol=1e-9)
            np.testing.assert_allclose(
                da.var(dim, ddof=1).values, np.nanvar(x, axis, ddof=1), rtol=1e-9
            )
            np.testing.assert_array_equal(da.count(dim).values, (~np.isnan(x)).sum(axis))
            np.testing.assert_allclose(da.sum(dim, skipna=False).values, x.sum(axis), rtol=1e-11)
            np.testing.assert_allclose(da.mean(dim, skipna=False).values, x.mean(axis), rtol=1e-11)
            enough = (~np.isnan(x)).sum(axis) >= 3
            want = np.where(enough, np.nansum(x, axis), np.nan)
            np.testing.assert_allclose(da.sum(dim, min_count=3).values, want, rtol=1e-11)


def test_integer_means_and_bools_through_the_kernels():
    x = np.arange(100_000, dtype=np.int32).reshape(100, 1000)
    da = ar.DataArray(x, dims=("a", "b"))
    np.testing.assert_allclose(da.mean().values, x.mean())
    np.testing.assert_allclose(da.mean("a").values, x.mean(0))
    np.testing.assert_allclose(da.std("b").values, x.std(1))
    flags = ar.DataArray(x % 2 == 0, dims=("a", "b"))
    assert flags.max().values == np.True_
    assert flags.min().values == np.False_
    np.testing.assert_allclose(flags.mean().values, 0.5)


REPRODUCIBLE = """
import numpy as np
import annoray as ar

x = np.random.default_rng(0).standard_normal(1_000_003)
x = x * 10.0 ** (np.arange(1_000_003) % 17 - 8)
da = ar.DataArray(x, dims="x")
grid = ar.DataArray(x[:1_000_000].reshape(1000, 1000), dims=("a", "b"))
out = [da.sum(), da.mean(), da.std(), grid.sum("a"), grid.mean("b"), grid.std("a")]
print(b"".join(o.values.tobytes() for o in out).hex())
"""


def test_results_do_not_depend_on_the_number_of_threads():
    def run(threads: str) -> str:
        done = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", REPRODUCIBLE],
            capture_output=True,
            encoding="utf-8",
            check=True,
            env={**os.environ, "RAYON_NUM_THREADS": threads},
        )
        return done.stdout.strip()

    first = run("1")
    assert first
    assert all(run(t) == first for t in ("2", "7", "16", "16"))
