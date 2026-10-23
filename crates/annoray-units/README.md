# annoray-units

Runtime and compile-time physical units with exact scales; the units layer of
[annoray](https://docs.rs/annoray), independent of it.

- **Exact scales:** scales are prime-factorised (`2^a · 3^b · 5^c · π^d · residual`), so SI
  prefixes and minute/hour/day/degree factors are exact and unit equality is tolerance-free.
  Conversion factors are the `f64` nearest the exact value (an inch is exactly `0.0254 m`),
  except where π enters.
- **Grammar:** UDUNITS-2/CF syntax backed by the complete UDUNITS-2 database, plus UCUM-style
  `kg.m2/s2`.
- **Unit systems:** SI, CGS (`dyn`, `erg`, `G`, `Mx`, `P`, `St`, `Gal`, `statC`, …; symbols take
  SI prefixes: `kG`, `cP`, `mGal`) and US customary/imperial (`ft`, `lb`, `oz`, `gallon`,
  `UK_gallon`, `psi`, `mph`, `nmi`, …). Customary units and physical constants use their legal
  or 2019-SI definitions (a US gallon is 231 in³, the electronvolt 1.602176634e-19 J) where the
  UDUNITS database rounds or predates them. Astronomy's and optics' angle and wavelength
  symbols parse too: `deg`, `arcmin`, `arcsec`, `mas`, `uas`, `AA`.
- **Non-multiplicative units:** affine (°C), epoch (`days since 1970-01-01`) and logarithmic
  (dB) units.
- **Instants and durations:** `Instant` and `Duration`, 64-bit nanosecond counts with the
  memory layout of numpy's `datetime64[ns]` and `timedelta64[ns]`.
- **Calendars:** all CF calendars (`noleap`, `360_day`, …) via `CfDate`, interconvertible with
  instants.
- **Dimension-typed quantities:** `Quantity<D>` with `typenum` exponents over the eight base
  dimensions (SI + angle); derived dimensions are composed with `dimension!`, and dimensional
  errors are type errors.

```rust
use annoray_units::{Unit, UnitPolicy};

let km: Unit = "km".parse()?;
let m: Unit = "m".parse()?;
assert_eq!(km.converter_to(&m)?.factor(), 1000.0);
assert!(km.plan_add(&m, UnitPolicy::Strict).is_err()); // scale mismatch under Strict

let c: Unit = "degC".parse()?;
assert_eq!(c.converter_to(&"K".parse()?)?.apply(20.0), 293.15);
# Ok::<(), annoray_units::UnitError>(())
```

```rust
use annoray_units::dims::{Length, Time};
use annoray_units::{Quantity, dimension};

dimension! {
    /// Rate of change of acceleration.
    pub Jerk = Length / Time^3;
}

let d = Quantity::<Length>::new(1.0, "m")?;
let t = Quantity::<Time>::new(2.0, "s")?;
let j: Quantity<Jerk> = (&(&(&d / &t)? / &t)? / &t)?; // L T^-3; any other exponent is a type error
assert_eq!(j.value(), 0.125);
# Ok::<(), annoray_units::UnitError>(())
```

## Compile-time unit libraries

The `uom` and `whippyunits` features implement `StaticUnit` for those libraries' quantity
types, giving `Quantity::to_static::<Q>()` / `Quantity::from_static(q)` conversions:

- **whippyunits** shares this crate's model (eight dimensions including angle;
  `2^a 3^b 5^c π^d` scales), so units map one to one and equal units convert without
  rescaling; only the default brand is bridged.
- **uom** stores coherent SI values and treats angle as dimensionless: conversions erase the
  angle dimension on both sides (degrees scale to radians), angle-kind quantities without a
  mass exponent come back in radians, `ThermodynamicTemperature` is absolute while
  `TemperatureInterval` converts affine sources by step only, and information quantities
  are rejected.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
