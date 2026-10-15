//! [`Quantity`]: a number with a unit, its dimension checked by the compiler (or, for
//! `Quantity<Dyn>`, at run time).

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::dims::{Dimensional, Prod, Quot};
use crate::{Unit, UnitError, UnitPolicy};

/// The dimension of a [`Quantity`] known only at run time (as for units read from files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dyn;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Dyn {}
    impl<D: crate::dims::Dimensional> Sealed for D {}
}

/// What a [`Quantity`]'s dimension is: a [`Dimensional`] type (checked by the compiler) or
/// [`Dyn`] (checked at run time).
pub trait Kind: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Whether `unit` can belong to a quantity of this kind.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if the unit's dimension is not this kind's.
    fn check(unit: &Unit) -> Result<(), UnitError>;
}

impl Kind for Dyn {
    fn check(_: &Unit) -> Result<(), UnitError> {
        Ok(())
    }
}

impl<D: Dimensional> Kind for D {
    fn check(unit: &Unit) -> Result<(), UnitError> {
        if unit.dim() == D::DIMENSION {
            Ok(())
        } else {
            Err(UnitError::Incompatible {
                lhs: unit.to_string(),
                rhs: D::DIMENSION.to_string(),
            })
        }
    }
}

/// Something that names a unit: a [`Unit`], or a string to parse.
pub trait IntoUnit {
    /// The unit.
    ///
    /// # Errors
    /// [`UnitError`] if a string does not parse.
    fn into_unit(self) -> Result<Unit, UnitError>;
}

impl IntoUnit for Unit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(self)
    }
}

impl IntoUnit for &Unit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(self.clone())
    }
}

impl IntoUnit for &str {
    fn into_unit(self) -> Result<Unit, UnitError> {
        self.parse()
    }
}

impl IntoUnit for &String {
    fn into_unit(self) -> Result<Unit, UnitError> {
        self.parse()
    }
}

/// A number with a unit.
///
/// `Quantity<Length>`, `Quantity<Velocity>`, …: the dimension is part of the type (see
/// [`dims`](crate::dims)), checked once when the quantity is made and by the compiler after
/// that: lengths add to lengths, and a length over a time *is* a velocity. The unit itself
/// (km or m, °C or K) is a runtime value, kept as declared, as in arrays: adding different
/// scales of the same dimension needs an explicit [`to`](Self::to), and absolute
/// temperatures neither add nor multiply. Operations that can meet those cases return
/// `Result`.
///
/// `Quantity<Dyn>` (the default) checks dimensions at run time, for units only known then.
/// Typed and dynamic quantities meet through [`dynamic`](Self::dynamic) and
/// [`cast`](Self::cast).
///
/// ```
/// use annoray_units::Quantity;
/// use annoray_units::dims::{Area, Length};
///
/// let w = Quantity::<Length>::new(3.0, "m")?;
/// let h = Quantity::<Length>::new(40.0, "cm")?;
/// let a: Quantity<Area> = (&w * &h.to("m")?)?;
/// assert_eq!(a.to_string(), "1.2000000000000002 m2");
/// assert!(Quantity::<Length>::new(1.0, "s").is_err());   // not a length
/// assert!((&w + &h).is_err());                            // m + cm: convert first
/// # Ok::<(), annoray_units::UnitError>(())
/// ```
pub struct Quantity<K: Kind = Dyn> {
    value: f64,
    unit: Unit,
    kind: PhantomData<K>,
}

impl<K: Kind> Clone for Quantity<K> {
    fn clone(&self) -> Self {
        Self::raw(self.value, self.unit.clone())
    }
}

impl<K: Kind> Quantity<K> {
    fn raw(value: f64, unit: Unit) -> Self {
        Self {
            value,
            unit,
            kind: PhantomData,
        }
    }

    /// `value` in `unit` (a [`Unit`] or a string).
    ///
    /// # Errors
    /// [`UnitError`] if the unit does not parse, or ([`UnitError::Incompatible`]) is not of
    /// this quantity's dimension.
    pub fn new(value: f64, unit: impl IntoUnit) -> Result<Self, UnitError> {
        let unit = unit.into_unit()?;
        K::check(&unit)?;
        Ok(Self::raw(value, unit))
    }

    /// The number, in [`unit`](Self::unit).
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// The unit.
    #[must_use]
    pub const fn unit(&self) -> &Unit {
        &self.unit
    }

    /// The value and the unit.
    #[must_use]
    pub fn into_parts(self) -> (f64, Unit) {
        (self.value, self.unit)
    }

    /// The same quantity in another unit of its dimension.
    ///
    /// # Errors
    /// [`UnitError`] if the unit does not parse or measures something else.
    pub fn to(&self, unit: impl IntoUnit) -> Result<Self, UnitError> {
        let unit = unit.into_unit()?;
        let c = self.unit.converter_to(&unit)?;
        Ok(Self::raw(c.apply(self.value), unit))
    }

    /// As a quantity whose dimension is checked at run time.
    #[must_use]
    pub fn dynamic(self) -> Quantity<Dyn> {
        Quantity::raw(self.value, self.unit)
    }

    /// As a quantity of dimension `D`, if its unit is one.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if the unit has another dimension.
    pub fn cast<D: Kind>(self) -> Result<Quantity<D>, UnitError> {
        D::check(&self.unit)?;
        Ok(Quantity::raw(self.value, self.unit))
    }

    /// `self^n` (dimension checked at run time; for a typed result, multiply).
    ///
    /// # Errors
    /// [`UnitError`] for non-linear units (absolute temperatures, …).
    pub fn powi(&self, n: i32) -> Result<Quantity<Dyn>, UnitError> {
        Ok(Quantity::raw(self.value.powi(n), self.unit.powi(n)?))
    }

    /// The `n`-th root (dimension checked at run time).
    ///
    /// # Errors
    /// [`UnitError::InexactRoot`] if the unit has no exact root.
    pub fn root(&self, n: i32) -> Result<Quantity<Dyn>, UnitError> {
        let unit = self.unit.root(n)?;
        Ok(Quantity::raw(self.value.powf(1.0 / f64::from(n)), unit))
    }

    /// The value in the dimensionless unit `1` (`5 %` is `0.05`).
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] unless the quantity is dimensionless.
    pub fn as_number(&self) -> Result<f64, UnitError> {
        Ok(self.unit.converter_to(&Unit::one())?.apply(self.value))
    }

    /// `self + rhs` or `self - rhs` under `policy` (`+`/`-` use the strict policy: the units
    /// must match).
    ///
    /// # Errors
    /// Unit errors: other dimensions, other scales (strict policy), absolute temperatures.
    pub fn add_with(&self, rhs: &Self, sub: bool, policy: UnitPolicy) -> Result<Self, UnitError> {
        let plan = if sub {
            self.unit.plan_sub(&rhs.unit, policy)?
        } else {
            self.unit.plan_add(&rhs.unit, policy)?
        };
        let a = plan
            .lhs
            .as_ref()
            .map_or(self.value, |c| c.apply(self.value));
        let b = plan.rhs.as_ref().map_or(rhs.value, |c| c.apply(rhs.value));
        Ok(Self::raw(if sub { a - b } else { a + b }, plan.result))
    }

    /// The same quantity, as kind `B` (only called where `K` and `B` are the same type).
    fn same<B: Kind>(self) -> Quantity<B> {
        Quantity::raw(self.value, self.unit)
    }

    fn product(&self, value: f64, unit: &Unit, div: bool) -> Result<(f64, Unit), UnitError> {
        Ok(if div {
            (self.value / value, self.unit.div(unit)?)
        } else {
            (self.value * value, self.unit.mul(unit)?)
        })
    }

    /// Compare with another quantity of the same kind, converting compatible units.
    /// `None` if the units are incompatible (possible for `Dyn`) or a value is NaN.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Option<Ordering> {
        let plan = self
            .unit
            .plan_cmp(&other.unit, UnitPolicy::AutoRescale)
            .ok()?;
        let a = plan
            .lhs
            .as_ref()
            .map_or(self.value, |c| c.apply(self.value));
        let b = plan
            .rhs
            .as_ref()
            .map_or(other.value, |c| c.apply(other.value));
        a.partial_cmp(&b)
    }
}

impl<K: Kind> PartialEq for Quantity<K> {
    /// Equal after converting compatible units (`1 km == 1000 m`).
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Some(Ordering::Equal)
    }
}

impl<K: Kind> PartialOrd for Quantity<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare(other)
    }
}

impl<K: Kind> fmt::Display for Quantity<K> {
    /// `2.5 km`; the format spec applies to the number (`{:.2}` gives `2.50 km`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)?;
        write!(f, " {}", self.unit)
    }
}

impl<K: Kind> fmt::Debug for Quantity<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({:?}, {:?})", self.value, self.unit.to_string())
    }
}

impl<K: Kind> Neg for Quantity<K> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::raw(-self.value, self.unit)
    }
}

impl<K: Kind> Neg for &Quantity<K> {
    type Output = Quantity<K>;
    fn neg(self) -> Quantity<K> {
        Quantity::raw(-self.value, self.unit.clone())
    }
}

/// Each operator for owned and borrowed operands alike: the borrowed form does the work.
macro_rules! forward {
    (impl[$($g:tt)*] $tr:ident $m:ident for $l:ty, $r:ty => $out:ty $(where $($w:tt)*)?) => {
        impl<$($g)*> $tr<$r> for $l $(where $($w)*)? {
            type Output = $out;
            fn $m(self, rhs: $r) -> $out {
                $tr::$m(&self, &rhs)
            }
        }
        impl<'a, $($g)*> $tr<&'a $r> for $l $(where $($w)*)? {
            type Output = $out;
            fn $m(self, rhs: &'a $r) -> $out {
                $tr::$m(&self, rhs)
            }
        }
        impl<'a, $($g)*> $tr<$r> for &'a $l $(where $($w)*)? {
            type Output = $out;
            fn $m(self, rhs: $r) -> $out {
                $tr::$m(self, &rhs)
            }
        }
    };
}

/// Quantities of kind `Self` and `Rhs` add and subtract: they have the same dimension.
/// (Implemented only for a kind with itself; the message explains a mismatch.)
#[diagnostic::on_unimplemented(
    message = "quantities of different dimensions do not add or subtract",
    label = "this has a different dimension from the other operand",
    note = "the dimensions are `{Self}` and `{Rhs}` (exponents of kg, m, s, A, K, mol, cd, rad)"
)]
pub trait SameDimension<Rhs> {}

impl<K: Kind> SameDimension<K> for K {}

impl<A: Kind + SameDimension<B>, B: Kind> Add<&Quantity<B>> for &Quantity<A> {
    type Output = Result<Quantity<A>, UnitError>;
    fn add(self, rhs: &Quantity<B>) -> Self::Output {
        self.add_with(&rhs.clone().same(), false, UnitPolicy::Strict)
    }
}

impl<A: Kind + SameDimension<B>, B: Kind> Sub<&Quantity<B>> for &Quantity<A> {
    type Output = Result<Quantity<A>, UnitError>;
    fn sub(self, rhs: &Quantity<B>) -> Self::Output {
        self.add_with(&rhs.clone().same(), true, UnitPolicy::Strict)
    }
}

forward!(impl[A: Kind + SameDimension<B>, B: Kind] Add add for Quantity<A>, Quantity<B> => Result<Quantity<A>, UnitError>);
forward!(impl[A: Kind + SameDimension<B>, B: Kind] Sub sub for Quantity<A>, Quantity<B> => Result<Quantity<A>, UnitError>);

// Typed products: the dimension of the result is computed by the compiler.
impl<A: Dimensional + Mul<B>, B: Dimensional> Mul<&Quantity<B>> for &Quantity<A>
where
    Prod<A, B>: Dimensional,
{
    type Output = Result<Quantity<Prod<A, B>>, UnitError>;
    fn mul(self, rhs: &Quantity<B>) -> Self::Output {
        let (v, u) = self.product(rhs.value, &rhs.unit, false)?;
        Ok(Quantity::raw(v, u))
    }
}

impl<A: Dimensional + Div<B>, B: Dimensional> Div<&Quantity<B>> for &Quantity<A>
where
    Quot<A, B>: Dimensional,
{
    type Output = Result<Quantity<Quot<A, B>>, UnitError>;
    fn div(self, rhs: &Quantity<B>) -> Self::Output {
        let (v, u) = self.product(rhs.value, &rhs.unit, true)?;
        Ok(Quantity::raw(v, u))
    }
}

forward!(impl[A: Dimensional + Mul<B>, B: Dimensional] Mul mul for Quantity<A>, Quantity<B>
    => Result<Quantity<Prod<A, B>>, UnitError> where Prod<A, B>: Dimensional);
forward!(impl[A: Dimensional + Div<B>, B: Dimensional] Div div for Quantity<A>, Quantity<B>
    => Result<Quantity<Quot<A, B>>, UnitError> where Quot<A, B>: Dimensional);

// Dynamic products (either side `Dyn`): dimension checked at run time.
macro_rules! dyn_products {
    ($(impl[$($g:tt)*] $l:ty, $r:ty;)*) => {$(
        impl<'a, 'b, $($g)*> Mul<&'b Quantity<$r>> for &'a Quantity<$l> {
            type Output = Result<Quantity<Dyn>, UnitError>;
            fn mul(self, rhs: &'b Quantity<$r>) -> Self::Output {
                let (v, u) = self.product(rhs.value, &rhs.unit, false)?;
                Ok(Quantity::raw(v, u))
            }
        }
        impl<'a, 'b, $($g)*> Div<&'b Quantity<$r>> for &'a Quantity<$l> {
            type Output = Result<Quantity<Dyn>, UnitError>;
            fn div(self, rhs: &'b Quantity<$r>) -> Self::Output {
                let (v, u) = self.product(rhs.value, &rhs.unit, true)?;
                Ok(Quantity::raw(v, u))
            }
        }
        forward!(impl[$($g)*] Mul mul for Quantity<$l>, Quantity<$r> => Result<Quantity<Dyn>, UnitError>);
        forward!(impl[$($g)*] Div div for Quantity<$l>, Quantity<$r> => Result<Quantity<Dyn>, UnitError>);
    )*};
}

dyn_products! {
    impl[] Dyn, Dyn;
    impl[D: Dimensional] D, Dyn;
    impl[D: Dimensional] Dyn, D;
}

// Plain numbers are dimensionless: scaling keeps the unit (an error for absolute
// temperatures, as for arrays).
impl<K: Kind> Mul<f64> for &Quantity<K> {
    type Output = Result<Quantity<K>, UnitError>;
    fn mul(self, rhs: f64) -> Self::Output {
        let (v, u) = self.product(rhs, &Unit::one(), false)?;
        Ok(Quantity::raw(v, u))
    }
}

impl<K: Kind> Div<f64> for &Quantity<K> {
    type Output = Result<Quantity<K>, UnitError>;
    fn div(self, rhs: f64) -> Self::Output {
        let (v, u) = self.product(rhs, &Unit::one(), true)?;
        Ok(Quantity::raw(v, u))
    }
}

impl<K: Kind> Mul<f64> for Quantity<K> {
    type Output = Result<Self, UnitError>;
    fn mul(self, rhs: f64) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl<K: Kind> Div<f64> for Quantity<K> {
    type Output = Result<Self, UnitError>;
    fn div(self, rhs: f64) -> Self::Output {
        Div::div(&self, rhs)
    }
}

impl<K: Kind> Mul<Quantity<K>> for f64 {
    type Output = Result<Quantity<K>, UnitError>;
    fn mul(self, rhs: Quantity<K>) -> Self::Output {
        Mul::mul(&rhs, self)
    }
}

impl<K: Kind> Mul<&Quantity<K>> for f64 {
    type Output = Result<Quantity<K>, UnitError>;
    fn mul(self, rhs: &Quantity<K>) -> Self::Output {
        Mul::mul(rhs, self)
    }
}
