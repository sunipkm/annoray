//! Conversions between runtime quantities and `uom` / `whippyunits` quantities.
#![cfg(any(feature = "uom", feature = "whippyunits"))]
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use annoray_units::{Dyn, Quantity, StaticUnit, Unit};

type Q = Quantity<Dyn>;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

fn unit(s: &str) -> Unit {
    s.parse().unwrap()
}

#[cfg(feature = "uom")]
mod uom_bridge {
    use super::*;
    use uom::si::angle::{degree, radian};
    use uom::si::f64::{
        Angle, AngularVelocity, Information, Length, Mass, TemperatureInterval,
        ThermodynamicTemperature, Torque, Velocity,
    };
    use uom::si::length::{kilometer, meter};
    use uom::si::thermodynamic_temperature::kelvin;

    #[test]
    fn units_are_coherent_si() {
        assert_eq!(Length::unit().unwrap(), unit("m"));
        assert_eq!(Mass::unit().unwrap(), unit("kg"));
        assert_eq!(Velocity::unit().unwrap(), unit("m s-1"));
        assert_eq!(Torque::unit().unwrap(), unit("N m")); // angle kind, but no angle
        assert_eq!(Angle::unit().unwrap(), unit("rad"));
        assert_eq!(AngularVelocity::unit().unwrap(), unit("rad s-1"));
        assert!(Information::unit().is_err());
    }

    #[test]
    fn quantities_round_trip() {
        let q = Q::new(2.5, "km").unwrap();
        let l: Length = q.to_static().unwrap();
        assert_eq!(l.get::<meter>(), 2500.0);
        let back = Quantity::from_static(Length::new::<kilometer>(1.5)).unwrap();
        assert_eq!(back.value(), 1500.0);
        assert_eq!(back.unit(), &unit("m"));
        assert!(Q::new(1.0, "s").unwrap().to_static::<Length>().is_err());
    }

    #[test]
    fn angles_are_dimensionless_in_uom() {
        let a: Angle = Q::new(180.0, "degree").unwrap().to_static().unwrap();
        assert!(close(a.get::<radian>(), std::f64::consts::PI));
        // A plain ratio converts too (uom's angle is dimensionless), in radians.
        let r: Angle = Q::new(0.5, "1").unwrap().to_static().unwrap();
        assert_eq!(r.get::<radian>(), 0.5);
        let back = Quantity::from_static(Angle::new::<degree>(90.0)).unwrap();
        assert!(close(back.to("degree").unwrap().value(), 90.0));
    }

    #[test]
    fn temperatures_absolute_and_intervals() {
        let c = Q::new(20.0, "degC").unwrap();
        let t: ThermodynamicTemperature = c.to_static().unwrap();
        assert!(close(t.get::<kelvin>(), 293.15));
        let d: TemperatureInterval = c.to_static().unwrap(); // a step: no offset
        assert!(close(d.value, 20.0));
    }
}

#[cfg(feature = "whippyunits")]
mod whippyunits_bridge {
    use super::*;
    use annoray_units::converter_into;
    use whippyunits::{qty, quantity};

    #[test]
    fn units_map_one_to_one() {
        assert_eq!(<qty!(km)>::unit().unwrap(), unit("km"));
        assert_eq!(<qty!(m / s)>::unit().unwrap(), unit("m s-1"));
        assert_eq!(<qty!(min)>::unit().unwrap(), unit("min"));
        assert_eq!(<qty!(deg)>::unit().unwrap(), unit("degree"));
        assert_eq!(<qty!(kg)>::unit().unwrap(), unit("kg"));
        // Identical scales: converting is the identity, no rescale.
        assert!(
            converter_into::<qty!(mm)>(&unit("mm"))
                .unwrap()
                .is_identity()
        );
    }

    #[test]
    fn quantities_round_trip() {
        let q = Q::new(1500.0, "m").unwrap();
        let k: qty!(km) = q.to_static().unwrap();
        assert_eq!(k.unsafe_value, 1.5);
        let back = Quantity::from_static(quantity!(3.0, min)).unwrap();
        assert_eq!(back.to("s").unwrap().value(), 180.0);
        // Angle is a real dimension on both sides.
        assert!(Q::new(1.0, "1").unwrap().to_static::<qty!(rad)>().is_err());
    }

    #[test]
    fn slices_view_in_place() {
        let raw = [1.0, 2.0, 3.0];
        let view = <qty!(km)>::view_slice(&raw).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view[1].unsafe_value, 2.0);
    }
}
