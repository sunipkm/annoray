# annoray-units

Runtime and compile-time physical units with exact scales; the units layer of
[annoray](https://docs.rs/annoray), independent of it.

- **Exact scales:** scales are prime-factorised (`2^a · 3^b · 5^c · π^d · residual`), so SI
  prefixes and minute/hour/day/degree factors are exact and unit equality is tolerance-free.
- **Grammar:** UDUNITS-2/CF syntax backed by the complete UDUNITS-2 database, plus UCUM-style
  `kg.m2/s2`.
- **Non-multiplicative units:** affine (°C), epoch (`days since 1970-01-01`) and logarithmic
  (dB) units.
- **Calendars:** all CF calendars (`noleap`, `360_day`, …) via `CfDate`, interconvertible with
  chrono.
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

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
