//! Typed quantities, from a crate that is not annoray's: dimensions compose across crates.
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::arithmetic_side_effects
)]

use annoray_units::dims::{
    Acceleration, Angle, AngularVelocity, Area, Dimensional, Energy, Force, Length, Mass,
    Temperature, Time, Velocity,
};
use annoray_units::{Dyn, Quantity, Unit, UnitError, dimension};

dimension! {
    /// Rate of change of acceleration: composed from annoray's dimensions.
    pub Jerk = Length / Time^3;
    /// And from our own.
    pub Snap = Jerk / Time;
    /// The same exponents as `Velocity`, under another name.
    pub Speed = Length * Time^-1;
    /// Torque: the dimension of energy.
    pub Torque = Force * Length;
    /// Specific energy, written the long way round.
    pub SpecificEnergy = Mass^-1 * Energy;
}

fn same_type<A: 'static, B: 'static>() -> bool {
    std::any::TypeId::of::<A>() == std::any::TypeId::of::<B>()
}

#[test]
fn dimensions_compose_as_types() {
    // Names that describe the same exponents are the same type.
    assert!(same_type::<Speed, Velocity>());
    assert!(same_type::<Torque, Energy>());
    assert!(!same_type::<Snap, Jerk>());
    assert_eq!(Jerk::DIMENSION.to_string(), "m s-3");
    assert_eq!(Snap::DIMENSION.to_string(), "m s-4");
    assert_eq!(SpecificEnergy::DIMENSION.to_string(), "m2 s-2");
    assert_eq!(Force::DIMENSION.to_string(), "kg m s-2");
    assert_eq!(AngularVelocity::DIMENSION.to_string(), "s-1 rad");
}

#[test]
fn typed_arithmetic() {
    let d = Quantity::<Length>::new(2.5, "km").unwrap();
    let t = Quantity::<Time>::new(30.0, "min").unwrap();
    // A length over a time is a velocity: the compiler knows.
    let v: Quantity<Velocity> = (&d / &t).unwrap();
    assert_eq!(v.unit().to_string(), "km min-1");
    let v = v.to("m/s").unwrap();
    assert!((v.value() - 2500.0 / 1800.0).abs() < 1e-12);
    let s = t.to("s").unwrap();
    let a: Quantity<Acceleration> = (&v / &s).unwrap();
    let j: Quantity<Jerk> = (&a / &s).unwrap();
    assert_eq!(j.unit().to_string(), "m s-3");
    // A speed is a velocity.
    let speed: Quantity<Speed> = v.clone();
    let back: Quantity<Length> = (&speed * &s).unwrap();
    assert!((back.to("km").unwrap().value() - 2.5).abs() < 1e-12);
    // Products of lengths are areas; owned operands work too.
    let area: Quantity<Area> = (d.clone() * d).unwrap();
    assert_eq!(area.unit().to_string(), "km2");
}

#[test]
fn units_are_checked_at_run_time() {
    // Not a length.
    assert!(matches!(
        Quantity::<Length>::new(1.0, "s"),
        Err(UnitError::Incompatible { .. })
    ));
    assert!(Quantity::<Length>::new(1.0, "no such unit").is_err());
    // Same dimension, different scales: convert first.
    let km = Quantity::<Length>::new(1.0, "km").unwrap();
    let m = Quantity::<Length>::new(1.0, "m").unwrap();
    assert!(matches!(&km + &m, Err(UnitError::ScaleMismatch { .. })));
    assert_eq!((&km + &m.to("km").unwrap()).unwrap().value(), 1.001);
    // Comparisons convert.
    assert!(km == Quantity::<Length>::new(1000.0, "m").unwrap());
    assert!(km > Quantity::<Length>::new(999.0, "m").unwrap());
    // Absolute temperatures: differences are fine, scaling is not.
    let c20 = Quantity::<Temperature>::new(20.0, "degC").unwrap();
    let c10 = Quantity::<Temperature>::new(10.0, "degC").unwrap();
    assert_eq!((&c20 - &c10).unwrap().value(), 10.0);
    assert!((&c20 * 2.0).is_err());
    assert_eq!(c20.to("degF").unwrap().value(), 68.0);
    // Angles are their own dimension.
    assert!(Quantity::<Angle>::new(90.0, "degree").is_ok());
    assert!(Quantity::<Angle>::new(90.0, "1").is_err());
    // Display and formatting.
    assert_eq!(format!("{km:.2}"), "1.00 km");
    assert_eq!(format!("{km:?}"), "Quantity(1.0, \"km\")");
}

#[test]
fn dynamic_quantities_meet_typed_ones() {
    // A unit read from a file: dimension known only at run time.
    let from_file: Quantity<Dyn> = Quantity::new(3.0, "m s-1").unwrap();
    let v: Quantity<Velocity> = from_file.clone().cast().unwrap();
    assert!(from_file.clone().cast::<Length>().is_err());
    let t = Quantity::<Time>::new(2.0, "s").unwrap();
    // Typed with dynamic gives dynamic.
    let d: Quantity<Dyn> = (&from_file * &t).unwrap();
    let d: Quantity<Length> = d.cast().unwrap();
    assert_eq!(d.value(), 6.0);
    let m: Unit = "m".parse().unwrap();
    assert_eq!((v.dynamic() * t.dynamic()).unwrap().unit(), &m);
    // Powers and roots, dimension checked at run time.
    let side = Quantity::<Length>::new(3.0, "m").unwrap();
    let sq: Quantity<Area> = side.powi(2).unwrap().cast().unwrap();
    assert_eq!(sq.root(2).unwrap().cast::<Length>().unwrap().value(), 3.0);
    // Numbers.
    assert_eq!(
        Quantity::<Dyn>::new(5.0, "%").unwrap().as_number().unwrap(),
        0.05
    );
    assert!(side.as_number().is_err());
    assert_eq!((-(2.0 * &side).unwrap()).value(), -6.0);
}
