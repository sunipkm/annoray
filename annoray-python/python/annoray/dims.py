"""Named dimensions: explicit, composable, and checked; the Python side of Rust's `dims`.

A dimension is the exponents of the base quantities (the seven SI ones and angle). The
names here are classes that stand for one. They:

- **make checked quantities:** ``Length(2.5, "km")`` is a ``Quantity[Length]``, and its unit
  must be a length (``Length(1, "s")`` raises ``UnitError``);
- **recognise them:** ``isinstance(q, Velocity)`` asks whether ``q`` (a ``Quantity`` or a
  ``DataArray``) has that dimension, however it was made: ``isinstance(d / t, Velocity)``;
- **compose:** ``Length / Time**3`` is a ``Dimension``; declare a name for it with a class::

      class Jerk(Dim, dimension=Length / Time**3):
          '''Rate of change of acceleration.'''

  Names that describe the same exponents are interchangeable, as in Rust.

Annotate with ``Quantity[Length]`` (or ``DataArray[Velocity]``), the spelling of Rust's
``Quantity<Length>``; type checkers accept it, and ``@checked`` enforces it at each call::

    @checked
    def speed(d: Quantity[Length], t: Quantity[Time]) -> Quantity[Velocity]:
        return d / t

Python checks when the code runs; the Rust twin (`annoray::dims`, `Quantity<D>`) checks when
it compiles.
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from annoray._annoray import DataArray, Dimension, Quantity, UnitError

__all__ = [
    "Acceleration",
    "Amount",
    "Angle",
    "AngularVelocity",
    "Area",
    "Charge",
    "Concentration",
    "Current",
    "Density",
    "Dim",
    "DimensionType",
    "Dimensionless",
    "Energy",
    "Force",
    "Frequency",
    "Length",
    "Luminosity",
    "Mass",
    "Momentum",
    "Power",
    "Pressure",
    "Resistance",
    "Temperature",
    "Time",
    "Velocity",
    "Voltage",
    "Volume",
    "checked",
    "dimension",
    "name_of",
]

# Dimension -> the first name given to it (for messages).
_names: dict[Dimension, str] = {}

D = TypeVar("D", bound="Dim")
F = TypeVar("F", bound=Callable[..., Any])


def _as_dimension(d: Dimension | DimensionType) -> Dimension:
    if isinstance(d, Dimension):
        return d
    if isinstance(d, DimensionType):
        return d.dimension
    raise TypeError(f"expected a dimension, not {d!r}")


class DimensionType(type):
    """The type of named dimensions (``Length``, ``Velocity``, your own ``Dim`` classes)."""

    dimension: Dimension

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        dimension: Dimension | DimensionType | None = None,
    ) -> DimensionType:
        cls = super().__new__(mcs, name, bases, namespace)
        if dimension is not None:
            d = _as_dimension(dimension)
            cls.dimension = d
            _names.setdefault(d, name)
        return cls

    def __call__(cls: type[D], magnitude: float, units: Any = None) -> Quantity[D]:  # type: ignore[misc]
        """A quantity of this dimension: ``Length(2.5, "km")``."""
        q: Quantity[D] = Quantity(magnitude, units)
        typing.cast("DimensionType", cls).check(q)
        return q

    def check(cls, value: Any) -> Any:
        """``value`` itself if it has this dimension (Rust's ``cast``); ``UnitError``
        otherwise."""
        if not cls.__instancecheck__(value):
            got = getattr(value, "dimension", None)
            raise UnitError(f"expected {cls.__name__} [{cls.dimension}], got {name_of(got) or got}")
        return value

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, (Quantity, DataArray)) and instance.dimension == cls.dimension

    # Named dimensions compose like the dimensions they stand for.
    def __mul__(cls, other: Dimension | DimensionType) -> Dimension:
        return cls.dimension * _as_dimension(other)

    def __rmul__(cls, other: Dimension | DimensionType) -> Dimension:
        return _as_dimension(other) * cls.dimension

    def __truediv__(cls, other: Dimension | DimensionType) -> Dimension:
        return cls.dimension / _as_dimension(other)

    def __rtruediv__(cls, other: Dimension | DimensionType | int) -> Dimension:
        if isinstance(other, int):
            return other / cls.dimension
        return _as_dimension(other) / cls.dimension

    def __pow__(cls, n: int) -> Dimension:
        return cls.dimension**n

    def __repr__(cls) -> str:
        return f"<dimension {cls.__name__} [{cls.dimension}]>"


class Dim(metaclass=DimensionType):
    """Base of named dimensions: ``class Jerk(Dim, dimension=Length / Time**3): ...``."""


def dimension(name: str, dim: Dimension | DimensionType, doc: str | None = None) -> DimensionType:
    """A named dimension made at run time. In code that type checkers read, prefer a class
    (``class Jerk(Dim, dimension=...)``): they accept classes in annotations."""
    d = _as_dimension(dim)
    return DimensionType(name, (Dim,), {"__doc__": doc or f"{name} [{d}]."}, dimension=d)


def name_of(d: object) -> str | None:
    """The name given to a dimension (the first one, if several), or ``None``."""
    return _names.get(d) if isinstance(d, Dimension) else None


def _dims_in(annotation: object) -> DimensionType | None:
    """The dimension an annotation names: ``Quantity[Length]``, ``DataArray[Length]``,
    ``Annotated[..., Length]`` or ``Length`` itself."""
    if isinstance(annotation, DimensionType):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (Quantity, DataArray, typing.Annotated):
        for arg in typing.get_args(annotation):
            if isinstance(arg, DimensionType):
                return arg
    return None


def checked(func: F) -> F:
    """Check dimension annotations of ``func``'s arguments and result at each call:
    ``def speed(d: Quantity[Length], t: Quantity[Time]) -> Quantity[Velocity]``. Other
    annotations are left alone."""
    sig = inspect.signature(func)
    # String annotations (`from __future__ import annotations`) may name dimensions imported
    # where the function is defined, not only at module level.
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    localns = dict(caller.f_locals) if caller is not None else None
    del frame, caller
    hints = typing.get_type_hints(func, localns=localns, include_extras=True)
    params = {k: d for k, v in hints.items() if k != "return" and (d := _dims_in(v))}
    result = _dims_in(hints.get("return"))

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        for name, dim in params.items():
            if name in bound.arguments:
                try:
                    dim.check(bound.arguments[name])
                except UnitError as e:
                    raise UnitError(f"{func.__name__}(): argument {name!r}: {e}") from None
        out = func(*args, **kwargs)
        if result is not None:
            try:
                result.check(out)
            except UnitError as e:
                raise UnitError(f"{func.__name__}(): result: {e}") from None
        return out

    return typing.cast("F", wrapper)


# ----- the base quantities -----


class Dimensionless(Dim, dimension=Dimension()):
    """No dimension (pure numbers, ratios)."""


class Mass(Dim, dimension=Dimension(mass=1)):
    """Mass (kg)."""


class Length(Dim, dimension=Dimension(length=1)):
    """Length (m)."""


class Time(Dim, dimension=Dimension(time=1)):
    """Time (s)."""


class Current(Dim, dimension=Dimension(current=1)):
    """Electric current (A)."""


class Temperature(Dim, dimension=Dimension(temperature=1)):
    """Thermodynamic temperature (K)."""


class Amount(Dim, dimension=Dimension(amount=1)):
    """Amount of substance (mol)."""


class Luminosity(Dim, dimension=Dimension(luminosity=1)):
    """Luminous intensity (cd)."""


class Angle(Dim, dimension=Dimension(angle=1)):
    """Plane angle (rad): a dimension of its own here, as in whippyunits."""


# ----- common derived dimensions (as Rust's `annoray::dims`) -----


class Frequency(Dim, dimension=1 / Time):
    """Frequency (s-1)."""


class Area(Dim, dimension=Length**2):
    """Area (m2)."""


class Volume(Dim, dimension=Length**3):
    """Volume (m3)."""


class Velocity(Dim, dimension=Length / Time):
    """Velocity (m s-1)."""


class Acceleration(Dim, dimension=Velocity / Time):
    """Acceleration (m s-2)."""


class Force(Dim, dimension=Mass * Acceleration):
    """Force (N = kg m s-2)."""


class Momentum(Dim, dimension=Mass * Velocity):
    """Momentum (kg m s-1)."""


class Energy(Dim, dimension=Force * Length):
    """Energy, work and heat (J = N m). Torque has the same dimension."""


class Power(Dim, dimension=Energy / Time):
    """Power (W = J s-1)."""


class Pressure(Dim, dimension=Force / Area):
    """Pressure and stress (Pa = N m-2)."""


class Density(Dim, dimension=Mass / Volume):
    """Density (kg m-3)."""


class Charge(Dim, dimension=Current * Time):
    """Electric charge (C = A s)."""


class Voltage(Dim, dimension=Power / Current):
    """Voltage (V = W A-1)."""


class Resistance(Dim, dimension=Voltage / Current):
    """Resistance (ohm = V A-1)."""


class AngularVelocity(Dim, dimension=Angle / Time):
    """Angular velocity (rad s-1)."""


class Concentration(Dim, dimension=Amount / Volume):
    """Concentration (mol m-3)."""
