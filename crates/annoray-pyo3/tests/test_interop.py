"""annoray-pyo3: annoray values through another extension module's Rust (the test extension
in tests/ext), as the annoray package's own objects.

Build the extension first (``maturin build -m crates/annoray-pyo3/tests/ext/Cargo.toml``)
and install it beside annoray.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import annoray as ar

ext = pytest.importorskip("annoray_pyo3_test")

NY = "America/New_York"


def same_variable(a: ar.DataArray, b: ar.DataArray) -> None:
    """The same dimensions, dtype, values (NaN equal), unit, time zone and attributes."""
    assert a.dims == b.dims
    assert a.dtype == b.dtype
    if a.dtype.kind in "fc":
        np.testing.assert_array_equal(a.values, b.values)
    elif a.dtype.kind in "Mm":  # NaT included
        np.testing.assert_array_equal(a.values.view("i8"), b.values.view("i8"))
    else:
        assert np.array_equal(np.asarray(a.values), np.asarray(b.values))
    assert a.units == b.units
    if a.dtype.kind == "M":
        assert a.dt.tz == b.dt.tz
    assert set(a.attrs) == set(b.attrs)
    for k, v in a.attrs.items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(b.attrs[k]))


def same_dataset(a: ar.Dataset, b: ar.Dataset) -> None:
    assert list(a.data_vars) == list(b.data_vars)
    assert list(a.coords) == list(b.coords)
    for k in [*a.data_vars, *a.coords]:
        same_variable(a[k], b[k])
    assert set(a.attrs) == set(b.attrs)
    for k, v in a.attrs.items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(b.attrs[k]))


# ----- the package check -----


def test_check():
    assert ext.check() == ar.__version__


@pytest.mark.parametrize(
    ("rust", "python", "ok"),
    [
        ("0.3.0", "0.3.4", True),  # 0.Y.Z: the same Y
        ("0.3.0-pre1", "0.3.0rc1", True),  # Cargo's and PEP 440's pre-releases
        ("0.3.1-dev.4+gabc", "0.3.1.dev4+gabc", True),
        ("0.3.0", "0.4.0", False),
        ("0.0.1", "0.0.2", False),  # 0.0.Z: the same Z
        ("0.0.0", "0.0.0", True),
        ("1.2.0", "1.9.3", True),  # X.Y.Z: the same X
        ("1.2.0", "2.0.0", False),
        ("0.3.0", "dev", False),
        ("1.2", "1.2", False),  # not X.Y.Z
    ],
)
def test_version_rule(rust, python, ok):
    assert ext.compatible(rust, python) is ok


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ("annoray.__version__ = '99.0.0'", "cannot exchange values with annoray 99.0.0"),
        ("annoray._annoray._interop_protocol = lambda: 999", "protocol 999"),
    ],
)
def test_mismatched_package_is_refused(patch, message):
    """In a fresh interpreter: the check runs once per process."""
    code = (
        "import annoray, annoray._annoray\n"
        f"{patch}\n"
        "import annoray_pyo3_test\n"
        "try:\n"
        "    annoray_pyo3_test.unit_roundtrip('m')\n"
        "except ImportError as e:\n"
        "    print('ImportError:', e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert out.stdout.startswith("ImportError:"), out
    assert message in out.stdout


# ----- units and quantities -----


@pytest.mark.parametrize("spelling", ["m", "km/h", "deg2", "degC", "W m-2 sr-1 um-1", "µm", "dB"])
def test_units(spelling):
    u = ar.Unit(spelling)
    for given in (u, spelling):
        back = ext.unit_roundtrip(given)
        assert isinstance(back, ar.Unit)
        assert back == u and str(back) == str(u)


def test_quantities():
    q = ar.Quantity(2.5, "km")
    back = ext.quantity_roundtrip(q)
    assert isinstance(back, ar.Quantity) and back == q and back.units == "km"
    assert ext.quantity_roundtrip("mm") == ar.Quantity(1.0, "mm")
    z = ar.Quantity(3 + 4j, "V")
    back = ext.complex_quantity_roundtrip(z)
    assert isinstance(back, ar.Quantity) and back == z
    # A real quantity widens to a complex one, not the reverse.
    assert ext.complex_quantity_roundtrip(ar.Quantity(2.0, "V")).magnitude == 2 + 0j
    with pytest.raises(TypeError, match="complex"):
        ext.quantity_roundtrip(z)
    # Typed dimensions are checked on the way in, and by the compiler inside.
    assert ext.length_roundtrip(ar.Quantity(3.0, "ft")) == ar.Quantity(3.0, "ft")
    with pytest.raises(ar.UnitError):
        ext.length_roundtrip(ar.Quantity(3.0, "s"))
    t = ext.travel_time(ar.Quantity(10.0, "km"), ar.Quantity(5.0, "km/h"))
    assert t.to("h") == ar.Quantity(2.0, "h")
    # annoray arithmetic on returned objects.
    assert (back * 2).to("mV").magnitude == 6000 + 8000j


def test_wrong_types_are_refused():
    with pytest.raises(TypeError, match="annoray.Unit or a unit string"):
        ext.unit_roundtrip(3.0)
    with pytest.raises(ar.UnitError):
        ext.unit_roundtrip("not a unit")
    with pytest.raises(TypeError, match="annoray.Quantity"):
        ext.quantity_roundtrip(2.5)
    with pytest.raises(TypeError, match="annoray.DataArray"):
        ext.dataarray_roundtrip(np.zeros(3))
    with pytest.raises(TypeError, match="annoray.DataArray"):
        ext.dataarray_roundtrip(ar.Dataset())
    with pytest.raises(TypeError, match="annoray.Dataset"):
        ext.dataset_roundtrip({"a": 1})
    with pytest.raises(TypeError, match="annoray.DataTree"):
        ext.tree_roundtrip(ar.Dataset())


# ----- data arrays, datasets, trees -----


def sample_array() -> ar.DataArray:
    times = np.datetime64("2024-07-01T02:00", "ns") + np.arange(3) * np.timedelta64(1, "h")
    return ar.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("time", "x"),
        coords={
            "time": ar.DataArray(times, dims="time").dt.tz_convert(NY),
            "x": ("x", [10.0, 20.0], {"units": "km", "long_name": "distance"}),
            "label": ("x", ["a", "ångström"]),
        },
        name="temp",
        units="degC",
        attrs={"source": "test", "scale": np.float32(0.5), "levels": np.array([1, 2], "i2")},
    )


def test_dataarray_round_trip():
    da = sample_array()
    for back in (
        ext.dataarray_roundtrip(da),
        ext.dataarray_by_reference(da),
        ext.dataarray_ref_roundtrip(da),
    ):
        assert isinstance(back, ar.DataArray)
        assert back.name == "temp"
        same_variable(back, da)
        assert list(back.coords) == list(da.coords)
        for k in da.coords:
            same_variable(back.coords[k], da.coords[k])
    back = ext.dataarray_roundtrip(da)
    # annoray's operations work on the result.
    assert (back.to("K") - da.to("K")).max().item() == ar.Quantity(0.0, "K")
    assert back.sel(x=20.0).sizes == {"time": 3}
    assert ext.convert(da, "K").units == "K"
    np.testing.assert_allclose(ext.convert(da, ar.Unit("degF")).values, da.to("degF").values)


def test_every_element_type_from_rust():
    ds = ext.make_samples()
    assert isinstance(ds, ar.Dataset)
    expected = {
        "f64": np.float64, "f32": np.float32, "c128": np.complex128, "c64": np.complex64,
        "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
        "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
        "bool": np.bool_, "duration": np.dtype("m8[ns]"),
    }  # fmt: skip
    for k, dtype in expected.items():
        assert ds[k].dtype == np.dtype(dtype), k
    assert ds["f64"].units == "degC" and ds["c128"].units == "V" and ds["scalar"].dims == ()
    assert ds["scalar"].item() == ar.Quantity(42.0, "hPa")
    assert list(ds["text"].values) == ["a", "ångström µm", ""]
    assert ds["i64"].values.tolist() == [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max]
    assert ds["u64"].values.tolist() == [0, 1, np.iinfo(np.uint64).max]
    time = ds.coords["time"]
    assert time.dt.tz == NY
    assert str(time.values[0]) == "2024-07-01T02:00:00.000000005" and np.isnat(time.values[1])
    assert ds.coords["x"].units == "km"
    np.testing.assert_array_equal(ds["duration"].values[[0, 2]], np.array([1, -2], "m8[h]"))
    assert ds.attrs["title"] == "samples" and list(ds.attrs["history"]) == ["one", "two"]
    # And back to Rust and Python again, unchanged; CF dates keep their nanoseconds (which
    # Python's cftime objects cannot show) through Python and back.
    same_dataset(ext.dataset_roundtrip(ds), ds)
    assert ext.noleap_nanoseconds(ext.dataset_roundtrip(ds)) == [7, None, 999_999_999]


def test_cf_dates():
    cftime = pytest.importorskip("cftime")
    ds = ext.make_samples()
    noleap = ds["noleap"].values
    assert noleap[0] == cftime.datetime(1, 2, 28, 1, 2, 3, 0, calendar="noleap")
    assert noleap[1] is None or (noleap[1] != noleap[1])
    back = ext.dataset_roundtrip(ds)
    assert list(back["noleap"].values[[0, 2]]) == list(noleap[[0, 2]])
    # CF dates made in Python.
    dates = ar.date_range("2000-01-01", periods=3, freq="D", calendar="360_day")
    da = ar.DataArray(np.arange(3.0), dims="t", coords={"t": dates})
    assert list(ext.dataarray_roundtrip(da).t.values) == list(da.t.values)


def test_tree_round_trip():
    tree = ar.DataTree.from_dict(
        {
            "/": ar.Dataset(coords={"x": ("x", [1.0, 2.0], {"units": "m"})}, attrs={"a": "root"}),
            "/surface": ar.Dataset({"t": ("x", [280.0, 281.0], {"units": "K"})}),
            "/upper/deep": ar.Dataset({"p": ((), 5.0, {"units": "hPa"})}),
        }
    )
    back = ext.tree_roundtrip(tree)
    assert isinstance(back, ar.DataTree)
    assert back.groups == tree.groups
    for path in tree.groups:
        same_dataset(
            back[path].to_dataset(inherit=False) if path != "/" else back.to_dataset(),
            tree[path].to_dataset(inherit=False) if path != "/" else tree.to_dataset(),
        )
    assert back.attrs == {"a": "root"}
    assert ext.tree_roundtrip(tree["surface"]).name == "surface"


# ----- copies -----

N = 10_000_000  # 80 MB of float64


def test_reading_in_place_is_zero_copy():
    da = ar.DataArray(np.arange(N, dtype="f8"), dims="x", units="m")
    values = da.values  # annoray's own memory, as a read-only view
    assert ext.view_address(da) == values.__array_interface__["data"][0]
    assert ext.view_sum(da) == float(values.sum())
    assert ext.describe(sample_array())[0] == ("temp", "number", "degC")
    assert ("time", "datetime", None) in ext.describe(sample_array())


COPIES = """
import resource, sys
import numpy as np
import annoray, annoray_pyo3_test as ext

def peak():  # the high-water mark of resident memory, in MiB
    if sys.platform == "linux":
        # This process image's own (ru_maxrss would include the parent's, kept across exec).
        with open("/proc/self/status", encoding="utf-8") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmHWM")) / 2**10
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
mb = {n} * 8 / 2**20
ext.check()
if "{direction}" == "to_python":
    base = peak()
    da = ext.make_big({n})  # made in Rust: the values, then annoray's copy
    assert isinstance(da, annoray.DataArray)
else:
    # The source kept alive, so nothing freed before `base` inflates the high-water mark.
    source = np.arange({n}, dtype="f8")
    da = annoray.DataArray(source, dims="x", units="m")
    base = peak()
    back = ext.dataarray_roundtrip(da)  # lent to Rust (no copy), copied there, back (one copy)
print((peak() - base) / mb)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="resource.getrusage is POSIX")
@pytest.mark.parametrize("direction", ["to_python", "round_trip"])
def test_large_arrays_are_copied_once_per_side(direction):
    """The peak memory a conversion adds, in units of the array's size, is 2: from Rust, the
    values made there and annoray's one copy of them; for a round trip, one copy into Rust
    (annoray lends its values) and one back into annoray. A third copy would make it 3."""
    out = subprocess.run(
        [sys.executable, "-c", COPIES.format(n=N, direction=direction)],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    copies = float(out.stdout.strip())
    assert 1.6 < copies < 2.5, copies
