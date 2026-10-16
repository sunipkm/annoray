"""head/tail/thin, drop_isel, drop_dims, reset_coords, filter_by_attrs, to_dataarray,
to_dataset(dim=), interp_like, reindex_like, combine_first, fillna, pad, rank, cumulative,
reduce, nbytes, indexes, chunk: against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def build(m):
    da = m.DataArray(
        np.array([[0.0, 1.0, np.nan], [3.0, 3.0, 5.0]]),
        dims=("x", "y"),
        coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0], "lab": ("y", ["a", "b", "c"])},
        name="a",
        attrs={"k": "v"},
    )
    ds = m.Dataset(
        {"a": da, "b": (("x",), np.array([1.0, 2.0]), {"units_note": "n"})}, attrs={"t": "g"}
    )
    return da, ds


def same(a, b, numeric_coords_only=False):
    assert tuple(a.dims) == tuple(b.dims)
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))
    for k in b.coords:
        if np.asarray(b.coords[k].values).dtype.kind in "OU":
            if numeric_coords_only:
                continue
            # Text padding: "" here, NaN objects in xarray.
            expected = [v if isinstance(v, str) else "" for v in np.asarray(b.coords[k].values)]
            assert list(a.coords[k].values) == expected
            continue
        np.testing.assert_array_equal(
            np.asarray(a.coords[k].values), np.asarray(b.coords[k].values)
        )


ARRAY_CASES = [
    lambda o: o.head(1),
    lambda o: o.head(y=2),
    lambda o: o.tail(y=1),
    lambda o: o.thin(y=2),
    lambda o: o.drop_isel(y=[0, 2]),
    lambda o: o.interp_like(o.isel(y=[0, 1])),
    lambda o: o.reindex_like(o.isel(y=[0, 2])),
    lambda o: o.combine_first(o.fillna(-1.0)),
    lambda o: o.fillna(o.isel(y=0)),
    lambda o: o.fillna(0.0),
    lambda o: o.pad(y=(1, 2)),
    lambda o: o.pad(y=1, mode="edge"),
    lambda o: o.pad(y=(2, 1), mode="reflect"),
    lambda o: o.pad(y=1, mode="wrap"),
    lambda o: o.pad(y=1, constant_values=0),
    lambda o: o.rank("y"),
    lambda o: o.rank("x", pct=True),
    lambda o: o.cumulative("y").sum(),
    lambda o: o.cumulative("y").mean(),
    lambda o: o.reduce(np.nanmax, "y"),
    lambda o: o.reduce(np.nansum),
]


@pytest.mark.parametrize("case", range(len(ARRAY_CASES)))
def test_array_methods(case):
    da, _ = build(ar)
    xda, _ = build(xr)
    f = ARRAY_CASES[case]
    # interp drops text coordinates along the interpolated dimension (xarray reindexes them).
    same(f(da), f(xda), numeric_coords_only=case == 5)


def test_dataset_methods():
    da, ds = build(ar)
    xda, xds = build(xr)
    for f in (
        lambda o: o.head(x=1),
        lambda o: o.drop_isel(x=0),
        lambda o: o.pad(x=1),
        lambda o: o.fillna(0.0),
        lambda o: o.combine_first(o.fillna(7.0)),
    ):
        r, x = f(ds), f(xds)
        for k in x.data_vars:
            same(r[k], x[k])
    assert list(ds.drop_dims("y").data_vars) == list(xds.drop_dims("y").data_vars)
    assert list(ds.filter_by_attrs(units_note="n").data_vars) == ["b"]
    assert list(ds.filter_by_attrs(k=lambda v: v == "v").data_vars) == ["a"]
    stacked, xstacked = ds.to_dataarray(), xds.to_dataarray()
    same(stacked, xstacked)
    assert list(stacked.coords["variable"].values) == ["a", "b"]
    split = da.to_dataset(dim="x")
    assert list(split.data_vars) == ["10.0", "20.0"]
    same(split["10.0"], xda.to_dataset(dim="x")[10.0])
    assert da.nbytes == xda.nbytes and ds.nbytes == xds.nbytes
    assert list(da.indexes) == list(xda.indexes)
    assert da.get_index("x").equals(xda.get_index("x"))
    assert da.chunk() is da and da.chunks is None


def test_reset_coords():
    da, ds = build(ar)
    xda, _ = build(xr)
    r, x = da.reset_coords(), xda.reset_coords()
    assert sorted(r.data_vars) == sorted(x.data_vars)
    assert sorted(da.reset_coords(drop=True).coords) == sorted(xda.reset_coords(drop=True).coords)
    with pytest.raises(ValueError, match="index"):
        ds.reset_coords("x")
    assert "lab" in ds.reset_coords("lab").data_vars


def test_units_in_methods():
    m = ar.DataArray([1.0, np.nan, 3.0], dims="t", coords={"t": [0.0, 1.0, 2.0]}, units="m")
    np.testing.assert_allclose(m.fillna(ar.Quantity(1, "km")).values, [1, 1000, 3])
    np.testing.assert_allclose(m.fillna(0).values, [1, 0, 3])
    with pytest.raises(ar.UnitError):
        m.fillna(5)
    km = ar.DataArray([2.0, 2.0, 2.0], dims="t", coords={"t": [0.0, 1.0, 2.0]}, units="km")
    np.testing.assert_allclose(m.fillna(km).values, [1, 2000, 3])
    np.testing.assert_allclose(m.combine_first(km).values, [1, 2000, 3])
    assert m.pad(t=1, constant_values=ar.Quantity(1, "km")).values[0] == 1000
    with pytest.raises(ar.UnitError):
        m.pad(t=1, constant_values=5)
    assert m.rank("t").units is None
    assert m.reduce(np.nansum, "t").units == "m"
    assert m.reduce(np.nanvar, "t", output_units="m2").units == "m2"
