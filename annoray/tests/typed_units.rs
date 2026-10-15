//! Arrays of uom / whippyunits quantities: conversion, zero-copy views, round trips, `sel`.
#![cfg(any(feature = "uom", feature = "whippyunits"))]
#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::indexing_slicing)]

use annoray::ndarray::{ArrayD, array};
use annoray::{DataArray, Unit, Variable, sel};

fn heights() -> DataArray {
    DataArray::from_array(["x"], array![1500.0, 250.0, 4000.0].into_dyn())
        .unwrap()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1000.0, 2000.0].into_dyn())
                .unwrap()
                .with_unit("m".parse().unwrap()),
        )
        .unwrap()
        .with_unit("m".parse().unwrap())
}

fn unit(s: &str) -> Unit {
    s.parse().unwrap()
}

#[cfg(feature = "whippyunits")]
mod whippyunits_arrays {
    use super::*;
    use annoray::CoreError;
    use whippyunits::{qty, quantity};

    #[test]
    fn convert_and_view() {
        let h = heights();
        let km = h.to_typed::<qty!(km)>().unwrap();
        assert_eq!(km[[0]].unsafe_value, 1.5);
        // Same unit: a view of the same memory.
        let v = h.view_typed::<qty!(m)>().unwrap();
        assert_eq!(v[[2]].unsafe_value, 4000.0);
        let raw = h.data::<f64>().unwrap();
        assert_eq!(v.as_ptr().cast::<f64>(), raw.as_ptr());
        // Another scale needs a conversion, not a view.
        assert!(matches!(
            h.view_typed::<qty!(km)>(),
            Err(CoreError::NotViewable(_))
        ));
        assert!(h.to_typed::<qty!(s)>().is_err());
    }

    type KmPerHour = qty!(km / h);

    #[test]
    fn round_trip() {
        let data: ArrayD<KmPerHour> =
            array![quantity!(36.0, km / h), quantity!(72.0, km / h)].into_dyn();
        let da = DataArray::from_typed(["t"], data.clone()).unwrap();
        let back = da.to_typed::<KmPerHour>().unwrap();
        assert_eq!(back, data);
        let ms = da.to_unit(&unit("m/s")).unwrap();
        assert_eq!(ms.to_f64().unwrap()[[0]], 10.0);
    }

    #[test]
    fn select_with_quantities() {
        let h = heights();
        let at = sel!(h[x = { quantity!(1.0, km) }]).unwrap();
        assert_eq!(at.to_f64().unwrap()[[]], 250.0);
    }
}

#[cfg(feature = "uom")]
mod uom_arrays {
    use super::*;
    use uom::si::angle::degree;
    use uom::si::f64::{Angle, Length, ThermodynamicTemperature, Velocity};
    use uom::si::length::{kilometer, meter};
    use uom::si::thermodynamic_temperature::kelvin;
    use uom::si::velocity::meter_per_second;

    #[test]
    fn convert_into_si() {
        let km = heights().to_unit(&unit("km")).unwrap();
        let l = km.to_typed::<Length>().unwrap();
        assert_eq!(l[[0]].get::<meter>(), 1500.0);
        // uom has no zero-copy view.
        assert!(km.view_typed::<Length>().is_err());
        let c = DataArray::from_array(["t"], array![20.0].into_dyn())
            .unwrap()
            .with_unit(unit("degC"));
        let t = c.to_typed::<ThermodynamicTemperature>().unwrap();
        assert!((t[[0]].get::<kelvin>() - 293.15).abs() < 1e-9);
        let deg = DataArray::from_array(["a"], array![180.0].into_dyn())
            .unwrap()
            .with_unit(unit("degree"));
        let a = deg.to_typed::<Angle>().unwrap();
        assert!((a[[0]].get::<degree>() - 180.0).abs() < 1e-9);
    }

    #[test]
    fn round_trip() {
        let v: ArrayD<Velocity> = array![
            Velocity::new::<meter_per_second>(3.0),
            Velocity::new::<meter_per_second>(4.5)
        ]
        .into_dyn();
        let da = DataArray::from_typed(["t"], v).unwrap();
        assert_eq!(da.unit(), Some(&unit("m s-1")));
        let kmh =
            DataArray::from_typed_in(["t"], &da.to_typed::<Velocity>().unwrap(), unit("km/h"))
                .unwrap();
        assert!((kmh.to_f64().unwrap()[[0]] - 10.8).abs() < 1e-9);
    }

    #[test]
    fn select_with_quantities() {
        let h = heights();
        let at = sel!(h[x = { Length::new::<kilometer>(2.0) }]).unwrap();
        assert_eq!(at.to_f64().unwrap()[[]], 4000.0);
    }
}
