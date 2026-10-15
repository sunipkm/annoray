//! [`StaticUnit`] for `uom` quantities of the SI system (feature `uom`).
//!
//! uom stores every value in coherent SI base units (m, kg, s, A, K, mol, cd) and treats angle
//! as dimensionless, separating angular quantities by a `Kind` marker. Accordingly:
//!
//! - the unit is the coherent SI unit of the quantity's dimension exponents;
//! - quantities of angle kind without a mass exponent (angle, angular velocity and
//!   acceleration, curvature) carry the angle dimension (radians), solid angles carry it
//!   squared (steradians); torque and angular momentum, also of angle kind in uom, do not;
//! - conversions ignore the angle dimension on both sides (degrees scale to radians);
//! - `ThermodynamicTemperature` is absolute (°C converts with its offset); every other
//!   temperature-bearing quantity, `TemperatureInterval` included, is a difference;
//! - information quantities (bits, bytes) have no counterpart and are an error.

use std::any::TypeId;

use uom::si::marker::{AngleKind, InformationKind, SolidAngleKind, TemperatureKind};
use uom::typenum::Integer;

use crate::static_unit::StaticUnit;
use crate::{Dimension, Scale, Unit, UnitError};

fn same<A: ?Sized + 'static, B: ?Sized + 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

macro_rules! bridge {
    ($($v:ty),*) => {$(
impl<D> StaticUnit for uom::si::Quantity<D, uom::si::SI<$v>, $v>
where
    D: uom::si::Dimension + ?Sized,
    D::Kind: 'static,
{
    type Value = $v;

    const ANGLE_DIMENSIONLESS: bool = true;

    fn unit() -> Result<Unit, UnitError> {
        if same::<D::Kind, dyn InformationKind>() {
            return Err(UnitError::Unsupported(
                "uom information quantities (bits, bytes) have no runtime unit".into(),
            ));
        }
        let mass = D::M::to_i8();
        let angle = if same::<D::Kind, dyn AngleKind>() && mass == 0 {
            1
        } else if same::<D::Kind, dyn SolidAngleKind>() {
            2
        } else {
            0
        };
        let dim = Dimension::from_exponents([
            mass,
            D::L::to_i8(),
            D::T::to_i8(),
            D::I::to_i8(),
            D::Th::to_i8(),
            D::N::to_i8(),
            D::J::to_i8(),
            angle,
        ]);
        Ok(Unit::linear(dim, Scale::ONE))
    }

    fn from_raw(v: $v) -> Self {
        Self {
            dimension: std::marker::PhantomData,
            units: std::marker::PhantomData,
            value: v,
        }
    }

    fn into_raw(self) -> $v {
        self.value
    }

    fn is_difference() -> bool {
        !same::<D::Kind, dyn TemperatureKind>()
    }
}
    )*};
}

bridge!(f32, f64);
