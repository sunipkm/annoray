"""Complex numbers from Python: element types, arithmetic and functions as numpy computes them,
units (both parts scaled; affine units refused), reductions, missing values, the operations
complex numbers have no meaning for (order; dropping the imaginary part), and storage."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

Z = np.array([1 + 2j, 3 - 1j, -0.5 + 0.25j])


def volts(values=Z):
    return ar.DataArray(values, dims="x", units="V")


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_element_types(dtype):
    da = ar.DataArray(Z.astype(dtype), dims="x")
    assert da.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(da.values, Z.astype(dtype))
    # complex64 stays complex64 with a float32 operand, as in numpy.
    out = da + ar.DataArray(np.ones(3, "float32"), dims="x")
    assert out.dtype == (np.ones(3, dtype) + np.ones(3, "float32")).dtype


def test_arithmetic_and_functions_match_numpy():
    da = ar.DataArray(Z, dims="x")
    other = np.array([2 - 1j, 0.5 + 0j, 1j])
    for got, want in [
        (da * ar.DataArray(other, dims="x"), Z * other),
        (da / ar.DataArray(other, dims="x"), Z / other),
        (da.conj(), np.conj(Z)),
        (np.sqrt(da), np.sqrt(Z)),
        (np.exp(da), np.exp(Z)),
    ]:
        np.testing.assert_allclose(got.values, want, rtol=1e-15)
    np.testing.assert_array_equal(da.real.values, np.real(Z))
    np.testing.assert_array_equal(da.imag.values, np.imag(Z))
    np.testing.assert_allclose(abs(da).values, np.abs(Z), rtol=1e-15)
    assert abs(da).dtype == np.float64


def test_units():
    da = volts()
    assert (da * 2.0).units == "V"
    assert abs(da).units == "V"
    assert da.real.units == "V" and da.imag.units == "V"
    assert (da * da).units == "V2"
    # Conversion scales both parts.
    mv = da.to("mV")
    assert mv.units == "mV"
    np.testing.assert_allclose(mv.values, Z * 1000.0, rtol=1e-15)
    assert volts(Z.astype("complex64")).to("mV").dtype == np.complex64
    # An offset has no meaning for a complex number.
    with pytest.raises(TypeError, match="affine"):
        ar.DataArray(Z, dims="x", units="degC").to("K")


def test_reductions():
    da = volts()
    for got, want in [(da.sum(), Z.sum()), (da.mean(), Z.mean())]:
        assert got.units == "V"
        assert got.values.item() == pytest.approx(want)
    # The variance of complex values is real (numpy's definition), in the unit squared.
    var = da.var()
    assert var.dtype == np.float64
    assert var.item().magnitude == pytest.approx(np.var(Z))
    assert var.units == "V2"


def test_missing_values():
    values = np.array([1 + 1j, complex(np.nan, 0.0), complex(0.0, np.nan)])
    da = ar.DataArray(values, dims="x")
    # A NaN in either part is missing, as numpy's isnan.
    np.testing.assert_array_equal(da.isnull().values, np.isnan(values))
    assert da.sum(skipna=True).item() == 1 + 1j


def test_where_and_fill():
    values = np.array([1 + 1j, complex(np.nan, 0.0), 3 - 2j])
    keep = ~np.isnan(values)
    da = ar.DataArray(values, dims="x")
    cond = ar.DataArray(keep, dims="x")
    for other in (np.full(3, 9j), np.full(3, 7.0), np.full(3, 7.0, "float32")):
        got = ar.where(cond, da, ar.DataArray(other, dims="x"))
        want = np.where(keep, values, other)
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got.values, want)
    single = da.astype("complex64")
    f32 = ar.DataArray(np.full(3, 7.0, "float32"), dims="x")
    assert ar.where(cond, single, f32).dtype == np.complex64
    np.testing.assert_array_equal(da.fillna(2 + 5j).values, np.where(keep, values, 2 + 5j))
    # As for real data, a number without a unit fills data with one only as 0 or NaN.
    v = ar.DataArray(values, dims="x", units="V")
    assert v.fillna(0j).units == "V"
    with pytest.raises(ar.UnitError):
        v.fillna(2 + 5j)
    # Complex data can be missing: not interpolating it is an error, not a no-op.
    with pytest.raises(TypeError, match="interpolate_na"):
        da.interpolate_na("x")


def test_no_order():
    """numpy orders complex numbers by real, then imaginary part; annoray refuses."""
    da = volts()
    with pytest.raises(TypeError, match="max"):
        da.max()
    with pytest.raises(TypeError, match="min"):
        da.min()
    with pytest.raises(TypeError):
        _ = da < ar.Quantity(1.0, "V")
    for op in ("argmax", "argmin", "median"):
        with pytest.raises(TypeError, match=op):
            getattr(da, op)()
    # numpy would rank (and sort) the real parts alone.
    with pytest.raises(TypeError, match="rank"):
        da.rank("x")
    with pytest.raises(ValueError, match="complex"):
        ar.DataArray(np.arange(3.0), dims="x", coords={"c": ("x", Z)}).sortby("c")
    # Equality is defined.
    np.testing.assert_array_equal((da == da.copy()).values, [True, True, True])


def test_no_silent_drop_of_the_imaginary_part():
    """numpy warns (ComplexWarning) and drops the imaginary part; annoray refuses."""
    with pytest.raises(ValueError, match="imaginary part"):
        ar.DataArray(Z, dims="x").astype("float64")
    # Values that are real convert exactly.
    real = ar.DataArray(np.array([1 + 0j, -2 + 0j]), dims="x").astype("float64")
    np.testing.assert_array_equal(real.values, [1.0, -2.0])


def test_storage(tmp_path):
    ds = ar.Dataset({"z": volts()})
    ds.to_netcdf(tmp_path / "z.nc")
    back = ar.open_dataset(tmp_path / "z.nc")
    np.testing.assert_array_equal(back["z"].values, Z)
    assert back["z"].units == "V"
    # A compound in netCDF-4/HDF5: read when the file is opened, not lazily.
    assert not back["z"].is_lazy
    for fmt in (2, 3):
        ds.to_zarr(tmp_path / f"z{fmt}.zarr", zarr_format=fmt)
        z = ar.open_zarr(tmp_path / f"z{fmt}.zarr")["z"]
        assert z.is_lazy
        np.testing.assert_array_equal(z.values, Z)
    with pytest.raises(OSError, match="NETCDF3|Netcdf3|netCDF-3|format"):
        ds.to_netcdf(tmp_path / "z3.nc", format="NETCDF3_64BIT")


# ----- complex quantities -----


def test_quantity_construction_and_parts():
    q = ar.Quantity(3 + 4j, "V")
    assert q.magnitude == 3 + 4j and isinstance(q.magnitude, complex)
    assert repr(q) == 'Quantity((3+4j), "V")'
    assert str(q) == "(3+4j) V" and f"{q:.1f}" == "3.0+4.0j V"
    assert q.real == ar.Quantity(3.0, "V") and q.imag == ar.Quantity(4.0, "V")
    assert q.conjugate().magnitude == 3 - 4j and q.conj() == q.conjugate()
    assert abs(q) == ar.Quantity(5.0, "V")
    # Real quantities keep a real magnitude, and have the same parts.
    r = ar.Quantity(2.0, "V")
    assert isinstance(r.magnitude, float)
    assert r.real == r and r.imag == ar.Quantity(0.0, "V") and r.conjugate() == r
    # A complex number times a unit, numpy's complex scalars included.
    assert 2j * ar.Unit("V") == ar.Quantity(2j, "V")
    assert np.complex64(1 + 2j) * ar.Unit("V") == ar.Quantity(1 + 2j, "V")
    with pytest.raises(TypeError, match="real or complex"):
        ar.Quantity("3", "V")


def test_quantity_arithmetic_and_conversion():
    q = ar.Quantity(3 + 4j, "V")
    assert q.to("mV") == ar.Quantity(3000 + 4000j, "mV")
    with pytest.raises(ar.UnitError, match="offset"):
        ar.Quantity(1 + 1j, "degC").to("K")
    assert (q + q).magnitude == 6 + 8j
    assert (q + ar.Quantity(1.0, "V")).magnitude == 4 + 4j  # a real operand widens
    with pytest.raises(ar.UnitError):
        _ = q + ar.Quantity(1j, "mV")  # different scales: convert first, as for real ones
    for got, want in [(q * 2, 6 + 8j), (2j * q, -8 + 6j), (q / 2j, 2 - 1.5j), (q**2, -7 + 24j)]:
        assert got.magnitude == want
    assert (q * 2).units == "V" and (2 * q).units == "V"  # a plain number keeps the spelling
    impedance = q / ar.Quantity(2j, "A")
    assert impedance.magnitude == 2 - 1.5j and impedance.units == "V A-1"
    assert (q**2).units == "V2"


def test_quantity_comparison():
    q = ar.Quantity(3 + 4j, "V")
    assert q == q.to("mV")
    assert q != ar.Quantity(3 - 4j, "V")
    assert ar.Quantity(2 + 0j, "V") == ar.Quantity(2.0, "V")
    assert q != ar.Quantity(3 + 4j, "m")  # other dimension: unequal
    with pytest.raises(TypeError, match="no order"):
        _ = q < q.conjugate()
    with pytest.raises(TypeError, match="not a float"):
        float(q)
    assert complex(ar.Quantity(50 + 25j, "%")) == 0.5 + 0.25j
    assert complex(ar.Quantity(50.0, "%")) == 0.5 + 0j


def test_quantity_with_arrays():
    z = volts(np.array([1 + 2j, 3 - 1j]))
    # A single value keeps its unit: a complex Quantity.
    assert z.isel(x=0).item() == ar.Quantity(1 + 2j, "V")
    assert z.sum().item() == ar.Quantity(4 + 1j, "V")
    assert z.real.sum().item() == ar.Quantity(4.0, "V")
    prod = z * ar.Quantity(1j, "A")
    np.testing.assert_array_equal(prod.values, np.array([1 + 2j, 3 - 1j]) * 1j)
    assert prod.units == "V A"
    keep = ar.DataArray([True, False], dims="x")
    np.testing.assert_allclose(
        z.where(keep, ar.Quantity(9j, "mV")).values, [1 + 2j, 0.009j], rtol=1e-15
    )
    gaps = volts(np.array([1 + 1j, complex(np.nan, 0.0)]))
    np.testing.assert_allclose(gaps.fillna(ar.Quantity(2j, "mV")).values, [1 + 1j, 0.002j])
    # Real data take a complex fill as numpy's where promotes.
    real = ar.DataArray([1.0, np.nan], dims="x", units="V").fillna(ar.Quantity(2j, "V"))
    np.testing.assert_array_equal(real.values, [1 + 0j, 2j])
    padded = z.pad(x=1, constant_values=ar.Quantity(500j, "mV"))
    np.testing.assert_allclose(padded.values, [0.5j, 1 + 2j, 3 - 1j, 0.5j])
    with pytest.raises(TypeError, match="cannot pad"):
        ar.DataArray([1.0], dims="x").pad(x=1, constant_values=1j)  # numpy would drop it
    with pytest.raises(TypeError, match="real fill"):
        ar.full_like(z, ar.Quantity(1j, "V"))


def test_quantity_astropy():
    u = pytest.importorskip("astropy.units")
    q = ar.Quantity(3 + 4j, "V")
    a = q.to_astropy()
    assert a.value == 3 + 4j and a.unit == u.V
    assert ar.Quantity((3 + 4j) * u.V) == q
    assert ar.from_astropy((3 + 4j) * u.V) == q
