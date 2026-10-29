"""String attributes: a one-element string array is a plain `str`, as in xarray and netCDF4-python."""

import numpy as np
import pytest

import annoray as ar

VALUES = {
    "one_ascii": ["abc"],
    "one_greek": ["Hα"],
    "two": ["a", "b"],
    "three": ["α", "", "βγ"],
    "plain": "Hα",
}


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "strings.nc"
    ar.DataArray(np.arange(3.0), dims="x", attrs=VALUES).to_netcdf(p)
    return p


def test_one_element_string_arrays_read_as_a_plain_str(path):
    attrs = ar.open_dataarray(path).attrs
    assert attrs["one_ascii"] == "abc"
    assert attrs["one_greek"] == "Hα"
    assert attrs["plain"] == "Hα"
    assert isinstance(attrs["one_greek"], str)


def test_longer_string_arrays_stay_lists(path):
    attrs = ar.open_dataarray(path).attrs
    assert attrs["two"] == ["a", "b"]
    assert attrs["three"] == ["α", "", "βγ"]


def test_dataset_variable_and_coordinate_attributes_follow_the_same_rule(tmp_path):
    ds = ar.Dataset(
        {"v": ("x", np.arange(2.0), {"names": ["only"], "many": ["a", "b"]})},
        coords={"x": ("x", [0.0, 1.0], {"label": ["Hβ"]})},
        attrs={"title": ["one"]},
    )
    p = tmp_path / "ds.nc"
    ds.to_netcdf(p)
    back = ar.open_dataset(p)
    assert back.attrs["title"] == "one"
    assert back["v"].attrs["names"] == "only"
    assert back["v"].attrs["many"] == ["a", "b"]
    assert back.coords["x"].attrs["label"] == "Hβ"


def test_matches_xarray_and_netcdf4(path):
    xr = pytest.importorskip("xarray")
    ours = ar.open_dataarray(path).attrs
    theirs = xr.open_dataarray(path).attrs
    for key in VALUES:
        assert ours[key] == theirs[key], key
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as f:
        var = next(iter(f.variables.values()))
        for key in VALUES:
            expected = var.getncattr(key)
            assert ours[key] == (
                list(expected) if isinstance(expected, (list, np.ndarray)) else expected
            ), key


def test_round_trip_through_a_second_write(path, tmp_path):
    first = ar.open_dataarray(path)
    second_path = tmp_path / "again.nc"
    first.to_netcdf(second_path)
    assert ar.open_dataarray(second_path).attrs["one_greek"] == "Hα"
