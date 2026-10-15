//! Quantities: numbers with units, their dimensions checked by the compiler.
//!
//! The Python twin (`annoray-python/examples/quantities.py`) does the same with quantities
//! checked at run time; both print the same lines.
//!
//! ```text
//! cargo run --example quantities
//! ```
// Quantity operators return `Result` (unit errors are values, never panics), which the
// workspace's `arithmetic_side_effects` lint cannot tell.
#![allow(clippy::arithmetic_side_effects)]

use std::error::Error;

use annoray::dims::{Area, Energy, Force, Length, Mass, Pressure, Temperature, Time, Velocity};
use annoray::{Dyn, Quantity, dimension};

dimension! {
    /// Rate of change of acceleration, composed from SI dimensions.
    pub Jerk = Length / Time^3;
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn main() -> Result<(), Box<dyn Error>> {
    // A length over a time is a velocity: the result's type says so.
    let d = Quantity::<Length>::new(2.5, "km")?;
    let t = Quantity::<Time>::new(30.0, "min")?;
    let v: Quantity<Velocity> = (&d / &t)?;
    println!("speed: {v:.3}");
    println!("speed: {:.3}", v.to("m/s")?);

    // Dimensions of your own, composed from the SI ones. A velocity over a time twice is a
    // jerk; dividing once more would not compile (it would be m s-4, not a jerk).
    let s = t.to("s")?;
    let j: Quantity<Jerk> = (&(&v.to("m/s")? / &s)? / &s)?;
    println!("jerk unit: {}", j.unit());
    println!("jerk: {:.3}", j.to("um s-3")?);

    // Physics reads like physics.
    let m = Quantity::<Mass>::new(1200.0, "kg")?;
    let car = Quantity::<Velocity>::new(27.8, "m/s")?;
    let e: Quantity<Energy> = ((&(&m * &car)? * &car)? * 0.5)?;
    println!("kinetic energy: {:.3}", e.to("kJ")?);
    let f = Quantity::<Force>::new(500.0, "N")?;
    let a = Quantity::<Area>::new(25.0, "cm2")?;
    let p: Quantity<Pressure> = (&f / &a)?;
    println!("pressure: {:.3}", p.to("kPa")?);

    // Units are values: same dimension, different scales need an explicit conversion.
    let km = Quantity::<Length>::new(1.0, "km")?;
    let one_m = Quantity::<Length>::new(1.0, "m")?;
    match &km + &one_m {
        Ok(_) => println!("1 km + 1 m -> (no error?)"),
        Err(e) => println!("1 km + 1 m -> error: {e}"),
    }
    println!("1 km + 1 m -> {:.3}", (&km + &one_m.to("km")?)?);
    println!(
        "1 km == 1000 m: {}",
        yes(km == Quantity::<Length>::new(1000.0, "m")?)
    );
    // (`km + t` does not compile: a length and a time do not add.)

    // Absolute temperatures: differences and conversions work; products do not.
    let warm = Quantity::<Temperature>::new(20.0, "degC")?;
    let cool = Quantity::<Temperature>::new(10.0, "degC")?;
    println!("20 degC - 10 degC = {:.1}", (&warm - &cool)?);
    println!("20 degC = {:.1}", warm.to("degF")?);
    println!("2 x 20 degC -> error: {}", yes((&warm * 2.0).is_err()));

    // A unit known only at run time (from a file, say): checked when cast.
    let read: Quantity<Dyn> = Quantity::new(12.0, "m s-1")?;
    println!(
        "is a velocity: {}",
        yes(read.clone().cast::<Velocity>().is_ok())
    );
    println!("is a length: {}", yes(read.cast::<Length>().is_ok()));
    Ok(())
}
