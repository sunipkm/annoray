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

## Unit

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

