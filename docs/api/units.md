---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Units

Units belong to the data: `da.units` is a {class}`Unit`, scalars with units are
{class}`Quantity` values, and a CF `units` attribute is decoded into a unit when a file is
opened. Dimensionally inconsistent operations raise {class}`UnitError`.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Unit
   Quantity
   Dimension
   UnitError
   dimension
```

Each section has a short example; the examples share this setup:

```{code-cell} python
import annoray as ar
```

## Unit

```{code-cell} python
u = ar.Unit("km/h")
print(u.dimension, u.si_factor, u.is_convertible("m/s"))
print(ar.Unit("degC").kind, ar.Unit("degC").offset)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Unit.dimension
   Unit.dimensionless
   Unit.is_convertible
   Unit.kind
   Unit.offset
   Unit.si_factor
   Unit.canonical
   Unit.to_astropy
```

## Quantity

```{code-cell} python
q = ar.Quantity(90.0, "km/h")
print(q.to("m/s"), q.magnitude, q.units)
try:
    q + ar.Quantity(1.0, "kg")
except ar.UnitError as e:
    print("UnitError:", e)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Quantity.magnitude
   Quantity.units
   Quantity.unit
   Quantity.dimension
   Quantity.to
   Quantity.to_astropy
```

## Dimension

```{code-cell} python
print(ar.Unit("N").dimension.exponents)
print(ar.Unit("m/m").dimension.dimensionless)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Dimension.exponents
   Dimension.dimensionless
```

## Dimension annotations

`annoray.dims` names dimensions as classes: they make checked quantities
(`Length(2.5, "km")`), annotate arrays and quantities (`DataArray[Velocity]`), and
{func}`annoray.dims.checked` enforces the annotations at call time. Derived dimensions are
declared as {class}`annoray.dims.Dim` subclasses.

```{code-cell} python
from annoray.dims import Length, Time, Velocity, checked

@checked
def speed(d: Length, t: Time) -> Velocity:
    return d / t

print(speed(ar.Quantity(5.0, "km"), ar.Quantity(30.0, "min")).to("km/h"))
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   dims.checked
   dims.Dim
   dims.name_of
```

```{eval-rst}
.. currentmodule:: annoray.dims

.. autosummary::
   :toctree: generated/

   Dimensionless
   Length
   Mass
   Time
   Current
   Temperature
   Amount
   Luminosity
   Angle
   Area
   Volume
   Velocity
   Acceleration
   Frequency
   AngularVelocity
   Force
   Pressure
   Energy
   Power
   Momentum
   Density
   Charge
   Voltage
   Resistance
   Concentration
```

