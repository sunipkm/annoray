"""Conversions between annoray and astropy units, quantities and arrays."""

from __future__ import annotations

import math

import numpy as np
import pytest

import annoray as ar

u = pytest.importorskip("astropy.units")
imperial = pytest.importorskip("astropy.units.imperial")


def same(a: ar.Unit, b: ar.Unit) -> bool:
    return a.dimension == b.dimension and math.isclose(a.si_factor, b.si_factor, rel_tol=1e-12)


@pytest.mark.parametrize(
    "text",
    [
        "km / s",
        "m s-1",
        "kg m2 s-2",
        "erg",
        "Angstrom",
        "hPa",
        "mm/h",
        "sr",
        "degree",
        "1",
        "%",
        "degK",
        "W m-2",
        "rad s-1",
        "yr",
        "min",
    ],
)
def test_units_round_trip(text):
    unit = ar.Unit(text)
    back = ar.Unit(unit.to_astropy())
    assert same(unit, back)
    decomposed = unit.to_astropy().decompose([u.kg, u.m, u.s, u.A, u.K, u.mol, u.cd, u.rad])
    assert math.isclose(decomposed.scale, unit.si_factor, rel_tol=1e-12)


def test_names_kept_where_they_agree():
    assert str(ar.Unit("km / s").to_astropy()) == "km / s"
    assert ar.Unit("degree").to_astropy() == u.deg
    assert str(ar.Unit(u.deg)) == "degree"  # astropy's deg is UDUNITS' degree, exactly
    assert str(ar.Unit(u.AU)) == "au"


def test_names_that_differ_fall_back_to_values():
    # UDUNITS' yr is the tropical year; astropy's the Julian year.
    assert ar.Unit(u.yr).si_factor == 365.25 * 86400
    assert ar.Unit("yr").to_astropy().decompose().scale == pytest.approx(31556925.9747)
    assert ar.Unit(u.Jy).dimension == ar.Dimension(mass=1, time=-2)


def test_angle_is_a_dimension_on_both_sides():
    assert ar.Unit(u.deg / u.s).dimension == ar.Dimension(angle=1, time=-1)
    assert ar.Unit(u.sr).dimension == ar.Dimension(angle=2)
    assert not ar.Unit(u.rad).dimensionless


def test_temperatures():
    assert ar.Unit("degC").to_astropy() == u.deg_C
    assert ar.Unit("degF").to_astropy() == imperial.deg_F
    assert ar.Quantity(20 * u.deg_C).to("K").magnitude == pytest.approx(293.15)
    assert ar.Quantity(32.0, "degF").to_astropy().to(u.deg_C, u.temperature()).value == (
        pytest.approx(0.0)
    )


def test_unrepresentable_units():
    with pytest.raises(ar.UnitError):
        ar.Unit("days since 2000-01-01").to_astropy()
    with pytest.raises(ar.UnitError):
        ar.Unit(u.dex)
    with pytest.raises(ar.UnitError):
        ar.Unit(u.m**0.5)


def test_quantities():
    q = ar.Quantity(3 * u.km / u.s)
    assert (q.magnitude, str(q.unit)) == (3.0, "km / s")
    back = q.to_astropy()
    assert back.unit == u.km / u.s and back.value == 3.0
    assert ar.from_astropy(2 * u.m) == ar.Quantity(2.0, "m")
    assert ar.to_astropy(ar.Quantity(5.0, "hPa")) == 500 * u.Pa


def test_arrays():
    q = np.arange(3.0) * u.W / u.m**2
    da = ar.from_astropy(q, dims="x", name="flux", coords={"x": [1.0, 2.0, 3.0]})
    assert isinstance(da, ar.DataArray)
    assert da.dims == ("x",) and da.name == "flux"
    np.testing.assert_array_equal(da.values, [0.0, 1.0, 2.0])
    assert da.unit is not None and same(da.unit, ar.Unit("W m-2"))
    back = da.to_astropy()
    assert back.unit.is_equivalent(u.W / u.m**2)
    np.testing.assert_array_equal(back.to_value(u.W / u.m**2), [0.0, 1.0, 2.0])
    bare = ar.DataArray(np.ones(2), dims="x").to_astropy()
    assert bare.unit == u.dimensionless_unscaled
    with pytest.raises(TypeError, match="from_astropy"):
        ar.Quantity(q)


def test_astropy_units_accepted_where_units_are():
    da = ar.DataArray(np.array([1.0, 2.0]), dims="x", units=u.km)
    assert str(da.units) == "km"
    assert da.to(u.m).values.tolist() == [1000.0, 2000.0]
    assert ar.Quantity(1.0, u.h).to("min").magnitude == 60.0
