---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Units and quantities

In annoray a physical unit is part of the data, not an attribute beside it. Every
{class}`~annoray.DataArray` carries a {class}`~annoray.Unit` (or none), single values with a
unit are {class}`~annoray.Quantity` objects, and every operation checks that the units of its
operands fit together. A CF `units` attribute is decoded into a unit when a file is opened and
written back when it is saved.

```{code-cell} python
import numpy as np
import annoray as ar
```

## Writing units

Units are written in the UDUNITS-2 grammar that the CF conventions use, so every `units`
attribute of a CF file parses as it is. The same unit has several spellings:

```{code-cell} python
same = ["m/s", "m s-1", "m.s-1", "m s^-1", "meters per second", "m.s**-1"]
print({ar.Unit(s) == ar.Unit("m/s") for s in same})
```

| Form | Examples |
|---|---|
| Symbols, with SI prefixes | `km`, `hPa`, `µm` or `um`, `GHz`, `mGal`, `kG` |
| Names, singular or plural, with prefixes | `meter`, `metres`, `kilogram`, `millibar` |
| Products: space, `.` or `*` | `kg m2 s-2`, `kg.m2.s-2`, `N*m` |
| Quotients | `W/m2`, `W m-2`, `mol/(m2 s)` |
| Powers: bare digits, `^` or `**` | `m2`, `m^2`, `m**2`, `s-1` |
| Numeric factors | `100 Pa`, `1e-3`, `0.01 m` |
| Offsets and epochs | `degC`, `°F`, `K @ 273.15`, `days since 2000-01-01` |
| Logarithmic | `dB`, `B` |
| Dimensionless | `1`, `""`, `%`, `ppm` |

A string that does not parse is an error when the unit is made, not later:

```{code-cell} python
try:
    ar.Unit("meters per fortnite")
except ar.UnitError as e:
    print("UnitError:", e)
```

## Units on data

```{code-cell} python
speed = ar.DataArray([4.0, 12.5, 30.0], dims="t", units="km/h")
print(speed.units, "→", speed.to("m/s").values)
```

`to` converts to any unit of the same dimension, exactly where the conversion factor is a
rational number (inches to centimetres is 127/50, not a rounded float); conversions that
involve π, such as degrees to radians, are as exact as a double allows. `float32` data stays
`float32`.

A file's CF `units` attribute becomes the unit of the variable it belongs to, and
`to_netcdf` writes the unit back as the attribute. Units are therefore never stale: there is
no attribute that can disagree with the data after a conversion.

## Quantities

```{code-cell} python
d = ar.Quantity(2.5, "km")
t = ar.Quantity(30, "min")
v = d / t
print(v, "=", v.to("m/s"))
print(ar.Quantity(1, "km") == ar.Quantity(1000, "m"), ar.Quantity(1, "km") > ar.Quantity(900, "m"))
```

A quantity is also made by multiplying a number by a unit (`2.5 * ar.Unit("km")`), and is
accepted wherever a value with a unit is expected: as an operand of a data array, as a label
in `sel`, as the value of `put`, `where` and `fillna`.

## Arithmetic rules

annoray refuses operations whose result would depend on an unstated assumption about units.

**Dimensions must match** for `+`, `-` and comparisons:

```{code-cell} python
try:
    speed + ar.Quantity(1.0, "m")
except ar.UnitError as e:
    print("UnitError:", e)
```

**Scales must match too.** Adding kilometres to metres is not guessed at: the result's unit
would have to be chosen silently. One operand is converted first:

```{code-cell} python
a = ar.Quantity(1.0, "km")
b = ar.Quantity(250.0, "m")
try:
    a + b
except ar.UnitError as e:
    print("UnitError:", e)
print(a + b.to("km"))
```

Comparisons convert compatible units (`1 km > 900 m` above); they only decide, they produce
no unit.

**`*`, `/` and powers combine units**: `km / min` above, `m s-1 * s` is `m`. Integer powers
and exact roots of dimensioned data work (`(m2) ** 0.5` is `m`); other powers need
dimensionless data.

**Temperatures and other offset units.** The difference of two absolute temperatures is a
temperature *step*, in kelvin-sized degrees; their sum has no meaning:

```{code-cell} python
print(ar.Quantity(20, "degC") - ar.Quantity(15, "degC"))
try:
    ar.Quantity(20, "degC") + ar.Quantity(15, "degC")
except ar.UnitError as e:
    print("UnitError:", e)
```

Products of offset units (`degC * m`) are errors: convert to an absolute scale (`K`) first.
Sums over a dimension (`sum`) refuse °C; `mean`, `min` and `max` keep it, and `std` is a step
in K.

**Logarithmic units** (`dB`, `B`) convert only between the same reference.

**Functions.** Transcendental functions (`np.exp`, `np.log`) need dimensionless input;
`np.sin`, `np.cos` and `np.tan` take angles (converted to radians) or plain numbers; inverse
trigonometric functions of data with a unit return radians. NumPy ufuncs without a unit rule
refuse data with a unit instead of dropping it.

## Plain numbers beside units

A number without a unit next to data with one is accepted only where its meaning does not
depend on the unit: `0` (for ratio-scale and logarithmic units) and NaN. Any other value is
given as a quantity, so that a threshold cannot silently be in the wrong unit:

```{code-cell} python
temp = ar.DataArray([281.2, 283.9, 290.5], dims="x", units="K")
print(temp.where(temp > ar.Quantity(10, "degC")).values)
try:
    temp > 10
except ar.UnitError as e:
    print("UnitError:", e)
```

The same rule applies to `where`, `fillna`, `clip`, `isin`, `put` and `query`.

## Unit systems

All of UDUNITS-2 is available. The tables below list the units most often needed in each
system with their usual symbols; every entry is checked against the units engine by the test
suite. Symbols take SI prefixes (`kG`, `cP`, `mJy`); names take them too (`kilogauss`).

### SI

The seven SI base units, the radian (a dimension of its own in annoray, as in astropy), the
derived units with special names, and the units accepted for use with SI.

| Unit | Symbol | Value |
|---|---|---|
| metre | `m` | base |
| kilogram | `kg` | base |
| second | `s` | base |
| ampere | `A` | base |
| kelvin | `K` | base |
| mole | `mol` | base |
| candela | `cd` | base |
| radian | `rad` | base |
| newton | `N` | `kg m s-2` |
| pascal | `Pa` | `N m-2` |
| joule | `J` | `N m` |
| watt | `W` | `J s-1` |
| coulomb | `C` | `A s` |
| volt | `V` | `W A-1` |
| ohm | `ohm` | `V A-1` |
| tesla | `T` | `V s m-2` |
| hertz | `Hz` | `s-1` |
| litre | `L` | `1e-3 m3` |
| tonne | `t` | `1000 kg` |
| minute, hour, day | `min`, `h`, `d` | `60 s`, `3600 s`, `86400 s` |
| degree (angle) | `degree` | `pi/180 rad` |
| electronvolt | `eV` | `1.602176634e-19 J` |
| bar | `bar` | `1e5 Pa` |

### CGS

Mechanical CGS units are exact multiples of SI units. CGS electromagnetic units are mapped
onto SI quantities by their standard correspondence (1 G = 10⁻⁴ T, 1 statC = 1/2997924580 C),
as UDUNITS and astropy do: values convert, but equations written in Gaussian form keep their
factors of 4π and *c* relative to their SI form.

| Unit | Symbol | Value |
|---|---|---|
| centimetre | `cm` | `0.01 m` |
| gram | `g` | `1e-3 kg` |
| dyne | `dyn` | `1e-5 N` |
| erg | `erg` | `1e-7 J` |
| barye | `Ba` | `0.1 Pa` |
| poise | `P` | `0.1 Pa s` |
| stokes | `St` | `1e-4 m2 s-1` |
| galileo (gal) | `Gal` | `0.01 m s-2` |
| kayser | `kayser` | `100 m-1` |
| gauss | `G` | `1e-4 T` |
| maxwell | `Mx` | `1e-8 Wb` |
| oersted | `Oe` | `250/pi A m-1` |
| statcoulomb (franklin) | `statC`, `Fr`, `esu` | `C/2997924580` |
| statampere | `statA` | `statC s-1` |
| statvolt | `statV` | `299.792458 V` |
| abampere (biot) | `abA`, `Bi` | `10 A` |
| abcoulomb | `abC` | `10 C` |
| abvolt | `abV` | `1e-8 V` |
| jansky (astronomy) | `Jy` | `1e-26 W m-2 Hz-1` |

### US customary and imperial

Values are the legal definitions (NIST SP 811 and Handbook 44), so conversions are exact: a US
gallon is 231 in³, a pound 0.45359237 kg. US and imperial (UK) volumes differ and have
separate names; the bare names are the US ones. Since 2023 the fathom, chain, rod and furlong
are based on the international foot; the US survey foot keeps its own name.

| Unit | Symbol | Value |
|---|---|---|
| inch | `in` | `0.0254 m` |
| foot | `ft` | `0.3048 m` |
| yard | `yd` | `0.9144 m` |
| mile | `mi` | `1609.344 m` |
| nautical mile | `nmi` | `1852 m` |
| fathom | `fathom` | `6 ft` |
| furlong | `furlong` | `660 ft` |
| US survey foot | `US_survey_foot` | `1200/3937 m` |
| acre | `acre` | `43560 ft2` |
| pound | `lb`, `lbm` | `0.45359237 kg` |
| ounce | `oz` | `lb/16` |
| troy ounce | `ozt` | `31.1034768 g` |
| grain | `gr` | `6.479891e-5 kg` |
| stone | `stone` | `14 lb` |
| short ton | `short_ton` | `2000 lb` |
| long ton | `long_ton` | `2240 lb` |
| US gallon | `gallon`, `US_gallon` | `231 in3` |
| US quart | `qt` | `gallon/4` |
| US pint | `pt` | `gallon/8` |
| US fluid ounce | `fl_oz`, `floz` | `gallon/128` |
| imperial gallon | `UK_gallon`, `imperial_gallon` | `4.54609 L` |
| imperial pint | `UK_pint`, `imperial_pint` | `UK_gallon/8` |
| imperial fluid ounce | `UK_fluid_ounce` | `UK_gallon/160` |
| pound-force | `lbf` | `lb 9.80665 m s-2` |
| pound per square inch | `psi` | `lbf in-2` |
| mile per hour | `mph` | `mi/h` |
| knot | `knot`, `kt` | `nmi/h` |
| degree Fahrenheit | `degF`, `°F` | `degR @ 459.67` |
| degree Rankine | `degR` | `K/1.8` |
| British thermal unit | `BTU` | `1055.05585262 J` |
| horsepower | `hp` | `550 ft lbf/s` |

### Other common units

Physical constants are the exact values of the 2019 SI (the electronvolt, the elementary
charge) or CODATA 2022's (the dalton); the light year and the parsec follow from the exact
astronomical unit and the Julian year.

| Unit | Symbol | Value |
|---|---|---|
| standard atmosphere | `atm` | `101325 Pa` |
| torr | `Torr` | `atm/760` |
| millimetre of mercury | `mmHg` | `133.322387415 Pa` |
| hectare | `hectare` | `1e4 m2` |
| ångström | `angstrom`, `Å`, `AA` | `1e-10 m` |
| degree (of arc) | `deg`, `degree`, `°` | `pi/180 rad` |
| arcminute | `arcmin` | `deg/60` |
| arcsecond | `arcsec` | `arcmin/60` |
| milliarcsecond | `mas` | `1e-3 arcsec` |
| microarcsecond | `uas`, `µas` | `1e-6 arcsec` |
| barn | `barn` | `1e-28 m2` |
| watt-hour | `Wh` | `3600 J` |
| calorie (international table) | `cal` | `4.1868 J` |
| elementary charge | `e` | `1.602176634e-19 C` |
| dalton (atomic mass unit) | `Da`, `u` | `1.66053906892e-27 kg` |
| astronomical unit | `au` | `149597870700 m` |
| light year | `ly` | `9460730472580800 m` |
| parsec | `pc` | `648000/pi au` |

### Symbols to read carefully

Some short symbols mean something other than a guess might suggest. The meanings below are
UDUNITS' (and so CF's) or the established ones:

| Written | Means | Not |
|---|---|---|
| `gal`, `Gal`, `mGal` | galileo, the CGS unit of acceleration (gravity anomalies) | gallon: `gallon`, `US_gallon`, `UK_gallon` |
| `oz` | avoirdupois ounce (mass) | fluid ounce: `fl_oz` |
| `pt` | US liquid pint | imperial pint: `UK_pint` |
| `nmi` | nautical mile | nano-mile |
| `mph`, `kph` | miles, kilometres per hour | milli-, kilo-phot |
| `t` | tonne (1000 kg) | short ton: `short_ton` |
| `G` | gauss | giga- (a prefix, as in `GHz`) |
| `P` | poise | peta- (a prefix, as in `PW`) |
| `Gs` | gigasecond | gauss: `G` |
| `yr`, `year` | tropical year (UDUNITS) | Julian year: `365.25 d` |
| `C` | coulomb | degree Celsius: `degC`, `°C` |

```{code-cell} python
for s in ["gal", "gallon", "oz", "fl_oz", "nmi", "mph", "kG"]:
    u = ar.Unit(s)
    print(f"{s:>7}: dimension {u.dimension}, {u.si_factor:g} in SI units")
```

## Time

Time units with an epoch (`days since 2000-01-01`, `hours since 1970-01-01 00:00`) are
decoded into dates in the variable's CF calendar (all CF calendars are supported). A time unit
without an epoch (`days`, `hours`) is a duration and stays a quantity;
{meth}`~annoray.DataArray.to` converts it like any other unit. `months` and `years` are
refused as epochs, as cftime refuses them: their length varies.

## Dimension annotations

`annoray.dims` names dimensions as classes. They make checked quantities, recognise them, and
annotate functions; {func}`annoray.dims.checked` enforces the annotations at every call:

```{code-cell} python
from annoray.dims import Length, Time, Velocity, Dim, checked
from annoray import Quantity

class Jerk(Dim, dimension=Length / Time**3):
    """Rate of change of acceleration."""

@checked
def mean_speed(d: Quantity[Length], t: Quantity[Time]) -> Quantity[Velocity]:
    return d / t

print(mean_speed(Length(42.195, "km"), Time(2.1, "h")).to("km/h"))
print(isinstance(ar.Quantity(3, "m s-3"), Jerk))
try:
    mean_speed(Time(1, "h"), Time(2.1, "h"))
except ar.UnitError as e:
    print("UnitError:", e)
```

In Rust the same names are types, and the check happens at compile time:

```rust
use annoray::dims::{Length, Time, Velocity};
use annoray::Quantity;

let d = Quantity::<Length>::new(2.5, "km")?;
let t = Quantity::<Time>::new(30.0, "min")?;
let v: Quantity<Velocity> = (&d / &t)?; // any other dimension is a type error
```

## Other unit libraries

- **astropy** (`annoray[astropy]`): `to_astropy()` on units, quantities and arrays, and
  {func}`annoray.from_astropy`. Units correspond through their SI-plus-radian decomposition,
  and keep their names where both libraries define a name the same way.
- **xarray and pint-xarray**: {meth}`~annoray.DataArray.to_xarray` writes the unit as the
  `units` attribute, as pint-xarray and cf-xarray expect.
- **Rust**: the `uom` and `whippyunits` features convert to and from those crates'
  compile-time quantities.
