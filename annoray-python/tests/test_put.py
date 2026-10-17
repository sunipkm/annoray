"""``DataArray.put`` / ``put_sel``: updated copies, checked against xarray's in-place
assignment where the two agree, and annoray's stricter rules where they do not."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import annoray as ar


def _pair() -> tuple[ar.DataArray, xr.DataArray]:
    data = np.arange(20.0, dtype="f4").reshape(5, 4)
    coords = {"x": [10.0, 20.0, 30.0, 40.0, 50.0], "y": list("abcd")}
    return (
        ar.DataArray(data, dims=("x", "y"), coords=coords, name="v"),
        xr.DataArray(data.copy(), dims=("x", "y"), coords=coords, name="v"),
    )


@pytest.mark.parametrize(
    "key",
    [
        {"x": 1},
        {"x": -1, "y": 2},
        {"x": slice(1, 4)},
        {"x": slice(None, None, -2)},
        {"x": [3, 0], "y": [1, 2]},
        {"y": np.array([True, False, True, False])},
    ],
)
def test_put_matches_xarray_assignment(key: dict) -> None:
    a, x = _pair()
    x[key] = -1.0
    got = a.put(-1.0, key)
    np.testing.assert_array_equal(got.values, x.values)
    assert got.dtype == x.dtype
    assert a.values[1, 0] == 4.0, "the original is unchanged"


def test_put_sel_matches_xarray_loc() -> None:
    a, x = _pair()
    x.loc[{"x": slice(20.0, 40.0), "y": "c"}] = 0.0
    np.testing.assert_array_equal(a.put_sel(0.0, x=slice(20.0, 40.0), y="c").values, x.values)
    x2 = xr.DataArray(np.arange(4.0), dims="t", coords={"t": [0.0, 1.0, 2.0, 3.0]})
    a2 = ar.DataArray(np.arange(4.0), dims="t", coords={"t": [0.0, 1.0, 2.0, 3.0]})
    x2.loc[{"t": 1.0}] = 9.0  # what method="nearest" picks for 1.2
    np.testing.assert_array_equal(a2.put_sel(9.0, t=1.2, method="nearest").values, x2.values)


def test_values_broadcast_by_name() -> None:
    a, x = _pair()
    row = ar.DataArray(np.array([1.0, 2.0, 3.0, 4.0], dtype="f4"), dims="y")
    x[{"x": [0, 2]}] = xr.DataArray(row.values, dims="y")
    np.testing.assert_array_equal(a.put(row, x=[0, 2]).values, x.values)
    # An unlabelled array has the selection's shape and dimension order.
    block = np.full((2, 4), 7.0)
    x[{"x": [1, 3]}] = block
    np.testing.assert_array_equal(a.put(row, x=[0, 2]).put(block, x=[1, 3]).values, x.values)
    with pytest.raises(ValueError, match="selection's shape"):
        a.put(np.zeros(4), x=[1, 3])


def test_dimension_coordinates_must_match() -> None:
    a, _ = _pair()
    v = ar.DataArray([1.0, 2.0], dims="x", coords={"x": [40.0, 10.0]})
    got = a.put(v, x=[3, 0], y=0)
    assert got.values[3, 0] == 1.0 and got.values[0, 0] == 2.0
    with pytest.raises(ValueError, match="differ"):
        a.put(v, x=[0, 3], y=0)


def test_units() -> None:
    k = ar.DataArray(np.array([1.0, 2.0, 3.0]), dims="x", attrs={"units": "K"})
    assert k.put(ar.Quantity(500, "mK"), x=0).values[0] == pytest.approx(0.5)
    assert k.put(ar.Quantity(0, "degC"), x=0).values[0] == pytest.approx(273.15)
    assert k.put(0, x=0).values[0] == 0.0  # the bare zero
    assert np.isnan(k.put(np.nan, x=0).values[0])
    with pytest.raises(ar.UnitError):
        k.put(1.0, x=0)
    with pytest.raises(ar.UnitError):
        k.put(ar.Quantity(1, "m"), x=0)
    plain = ar.DataArray(np.array([1.0, 2.0]), dims="x")
    with pytest.raises(ar.UnitError):
        plain.put(ar.Quantity(1, "m"), x=0)
    assert str(k.put(ar.Quantity(1, "mK"), x=1).units) == "K"


def test_element_type_never_changes() -> None:
    i = ar.DataArray(np.array([1, 2, 3], dtype="u1"), dims="t")
    assert i.put(2.0, t=0).dtype == np.uint8
    for bad in (2.5, 300, -1, np.nan):
        with pytest.raises(ValueError, match="cannot convert"):
            i.put(bad, t=0)
    with pytest.raises((TypeError, ValueError)):
        i.put(True, t=0)
    f = ar.DataArray(np.zeros(2, dtype="f4"), dims="t")
    assert f.put(np.float64(0.1), t=0).dtype == np.float32
    s = ar.DataArray(np.array(["a", "b"]), dims="t")
    assert list(s.put("zz", t=1).values) == ["a", "zz"]


def test_positions_are_checked() -> None:
    a, _ = _pair()
    with pytest.raises(ValueError, match="twice"):
        a.put(0.0, x=[1, 1])
    with pytest.raises(IndexError):
        a.put(0.0, x=9)
    with pytest.raises(ValueError, match="not a dimension"):
        a.put(0.0, z=0)


def test_lazy_data_is_read_not_cached(tmp_path) -> None:
    a, _ = _pair()
    path = tmp_path / "v.nc"
    a.to_dataset().to_netcdf(path)
    lazy = ar.open_dataset(path)["v"]
    got = lazy.put(-5.0, x=0)
    assert got.values[0, 0] == -5.0
    assert lazy.values[0, 0] == 0.0


def test_assignment_points_to_put() -> None:
    a, _ = _pair()
    with pytest.raises(TypeError, match="put"):
        a[0] = 1.0  # type: ignore[index]  # refused at run time
    with pytest.raises(TypeError, match="put_sel"):
        a.loc[{"x": 10.0}] = 1.0
