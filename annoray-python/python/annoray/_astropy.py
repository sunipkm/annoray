"""Conversions between annoray and astropy units, quantities and arrays.

Both systems treat angle as a dimension, so units correspond through their decomposition into
the SI base units plus the radian. A unit keeps its name when the other side parses that name
as the same unit (``km / s``, ``erg``, ``Angstrom``); names that mean different units on the two
sides (UDUNITS' ``yr`` is the tropical year, astropy's the Julian year) are detected by that
check and fall back to the decomposed form (``31557600 s``). °C and °F map to astropy's
``deg_C`` and ``deg_F``; other non-multiplicative units (epochs, logarithmic units) have no
counterpart and raise :class:`UnitError`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from annoray._annoray import DataArray, Dimension, Quantity, Unit, UnitError

if TYPE_CHECKING:
    import astropy.units as au

#: annoray's base dimensions, their astropy units and their UDUNITS symbols.
_BASES = (
    ("mass", "kg"),
    ("length", "m"),
    ("time", "s"),
    ("current", "A"),
    ("temperature", "K"),
    ("amount", "mol"),
    ("luminosity", "cd"),
    ("angle", "rad"),
)


#: astropy names whose UDUNITS spelling differs but which mean exactly the same unit.
_RENAMES = {
    "deg": "degree",
    "arcmin": "arcminute",
    "arcsec": "arcsecond",
    "mas": "milliarcsecond",
    "AU": "au",
}
_RENAME = re.compile(r"\b(" + "|".join(_RENAMES) + r")\b")


def _au() -> Any:
    try:
        import astropy.units as au
    except ImportError as e:  # pragma: no cover - exercised only without astropy
        raise ImportError(
            'astropy conversions require astropy: pip install "annoray[astropy]"'
        ) from e
    return au


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def _decompose(unit: au.UnitBase) -> tuple[float, dict[str, int]]:
    """Scale and base-dimension exponents of an astropy unit."""
    a = _au()
    bases = [getattr(a, sym) for _, sym in _BASES]
    try:
        d = unit.decompose(bases=bases)
    except (a.UnitsError, TypeError) as e:
        raise UnitError(f"{unit} has no annoray equivalent ({e})") from None
    names = {getattr(a, sym): name for name, sym in _BASES}
    exponents: dict[str, int] = {}
    for base, power in zip(d.bases, d.powers, strict=True):
        if power != int(power):
            raise UnitError(f"{unit} has a fractional exponent ({base}^{power})")
        exponents[names[base]] = int(power)
    return float(cast("float", d.scale)), exponents


def _matches(unit: Unit, scale: float, exponents: Mapping[str, int]) -> bool:
    return (
        unit.kind == "linear"
        and unit.dimension == Dimension(**exponents)
        and _close(unit.si_factor, scale)
    )


def unit_from_astropy(unit: Any) -> Unit:
    """The annoray unit of an astropy unit (by name where both agree, else decomposed)."""
    a = _au()
    from astropy.units import imperial

    if unit == a.deg_C:
        return Unit("degC")
    if unit == imperial.deg_F:
        return Unit("degF")
    if isinstance(unit, a.UnrecognizedUnit):
        raise UnitError(f"astropy does not recognise the unit {unit}")
    scale, exponents = _decompose(unit)
    texts = []
    for fmt in ("generic", "fits", "cds"):
        try:
            texts.append(unit.to_string(fmt))
        except (ValueError, a.UnitsError):  # not expressible in this format
            continue
    texts += [_RENAME.sub(lambda m: _RENAMES[m.group(1)], t) for t in texts]
    for text in dict.fromkeys(texts):
        try:
            candidate = Unit(text or "1")
        except (UnitError, ValueError):
            continue
        if _matches(candidate, scale, exponents):
            return candidate
    parts = [
        f"{sym}{exponents[name]}" if exponents[name] != 1 else sym
        for name, sym in _BASES
        if exponents.get(name)
    ]
    prefix = "" if scale == 1.0 else f"{scale!r} "
    return Unit((prefix + " ".join(parts)).strip() or "1")


def unit_to_astropy(unit: Unit) -> Any:
    """The astropy unit of an annoray unit (by name where both agree, else composed)."""
    a = _au()
    from astropy.units import imperial

    if unit.kind == "affine":
        offset = unit.offset or 0.0
        if _close(unit.si_factor, 1.0) and _close(offset, 273.15):
            return a.deg_C
        if _close(unit.si_factor, 5 / 9) and _close(offset, 459.67):
            return imperial.deg_F
        raise UnitError(f"the absolute temperature unit {unit} has no astropy equivalent")
    if unit.kind != "linear":
        raise UnitError(f"the {unit.kind} unit {unit} has no astropy equivalent")
    scale = unit.si_factor
    exponents = unit.dimension.exponents
    for text in dict.fromkeys((str(unit), unit.canonical)):
        for fmt in ("generic", "fits", "cds"):
            candidate = a.Unit(text, format=fmt, parse_strict="silent")
            if isinstance(candidate, a.UnrecognizedUnit):
                continue
            try:
                got_scale, got = _decompose(candidate)
            except UnitError:
                continue
            if got == {k: v for k, v in exponents.items() if v} and _close(got_scale, scale):
                return candidate
    used = [(getattr(a, sym), exponents[name]) for name, sym in _BASES if exponents.get(name)]
    return a.CompositeUnit(scale, [b for b, _ in used], [p for _, p in used])


def to_astropy(obj: Unit | Quantity | DataArray) -> Any:
    """An annoray unit, quantity or array as its astropy counterpart (arrays lose their
    coordinates and attributes)."""
    a = _au()
    if isinstance(obj, Unit):
        return unit_to_astropy(obj)
    if isinstance(obj, Quantity):
        return a.Quantity(obj.magnitude, unit_to_astropy(obj.unit))
    if isinstance(obj, DataArray):
        unit = unit_to_astropy(obj.unit) if obj.unit is not None else a.dimensionless_unscaled
        return a.Quantity(np.asarray(obj.values), unit, copy=False)
    raise TypeError(f"expected an annoray Unit, Quantity or DataArray, not {type(obj).__name__}")


def quantity_parts(q: Any) -> tuple[float, Unit]:
    """Value and annoray unit of a scalar astropy quantity."""
    if not q.isscalar:
        raise TypeError("an array-valued astropy Quantity becomes a DataArray: use from_astropy")
    return float(q.value), unit_from_astropy(q.unit)


def from_astropy(
    obj: Any,
    dims: str | Sequence[str] | None = None,
    coords: Mapping[str, Any] | None = None,
    name: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Unit | Quantity | DataArray:
    """An astropy unit or quantity as annoray's: a :class:`Unit`, a :class:`Quantity` for
    scalars, a :class:`DataArray` (with ``dims``, ``coords``, ...) for arrays."""
    a = _au()
    if isinstance(obj, a.UnitBase):
        return unit_from_astropy(obj)
    if isinstance(obj, a.Quantity):
        if obj.isscalar and dims is None and coords is None:
            value, unit = quantity_parts(obj)
            return Quantity(value, unit)
        return DataArray(
            np.asarray(obj.value),
            dims=dims,  # type: ignore[arg-type]
            coords=coords,
            name=name,
            attrs=attrs,
            units=unit_from_astropy(obj.unit),
        )
    raise TypeError(f"expected an astropy unit or Quantity, not {type(obj).__name__}")


__all__ = ["from_astropy", "to_astropy", "unit_from_astropy", "unit_to_astropy"]
