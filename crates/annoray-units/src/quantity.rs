//! [`Quantity`]: a number with a unit, its dimension checked by the compiler (or, for
//! `Quantity<Dyn>`, at run time); the number is real (`f64`) or complex (`Complex<f64>`).

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::Complex;

use crate::dims::{Dimensional, Prod, Quot};
use crate::{Converter, Unit, UnitError, UnitId, UnitPolicy};

/// The dimension of a [`Quantity`] known only at run time (as for units read from files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dyn;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Dyn {}
    impl<D: crate::dims::Dimensional> Sealed for D {}

    pub trait SealedScalar {}
    impl SealedScalar for f64 {}
    impl SealedScalar for num_complex::Complex<f64> {}
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

/// The number a [`Quantity`] holds: `f64`, or `Complex<f64>` for a complex quantity.
pub trait Scalar:
    sealed::SealedScalar + Copy + PartialEq + fmt::Display + fmt::Debug + Send + Sync + 'static
{
    /// The value converted by `c`.
    ///
    /// # Errors
    /// [`UnitError::ComplexOffset`] for a complex value and a conversion with an offset.
    fn convert(self, c: &Converter, from: &Unit, to: &Unit) -> Result<Self, UnitError>;

    /// `self + other` (IEEE arithmetic, as every operation here: it never panics).
    #[must_use]
    fn plus(self, other: Self) -> Self;

    /// `self - other`.
    #[must_use]
    fn minus(self, other: Self) -> Self;

    /// `self * other`.
    #[must_use]
    fn times(self, other: Self) -> Self;

    /// `self / other`.
    #[must_use]
    fn over(self, other: Self) -> Self;

    /// `-self`.
    #[must_use]
    fn negated(self) -> Self;

    /// `self^n`.
    #[must_use]
    fn powi(self, n: i32) -> Self;

    /// The principal `n`-th root.
    #[must_use]
    fn root(self, n: i32) -> Self;
}

/// The arithmetic of a [`Scalar`] type, by its operators.
macro_rules! scalar_arithmetic {
    () => {
        fn plus(self, other: Self) -> Self {
            self + other
        }
        fn minus(self, other: Self) -> Self {
            self - other
        }
        fn times(self, other: Self) -> Self {
            self * other
        }
        fn over(self, other: Self) -> Self {
            self / other
        }
        fn negated(self) -> Self {
            -self
        }
    };
}

impl Scalar for f64 {
    scalar_arithmetic!();
    fn convert(self, c: &Converter, _: &Unit, _: &Unit) -> Result<Self, UnitError> {
        Ok(c.apply(self))
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    fn root(self, n: i32) -> Self {
        self.powf(1.0 / f64::from(n))
    }
}

impl Scalar for Complex<f64> {
    scalar_arithmetic!();
    /// Both parts scaled; an offset (°C ↔ K) has no meaning for a complex number.
    fn convert(self, c: &Converter, from: &Unit, to: &Unit) -> Result<Self, UnitError> {
        if c.offset() != 0.0 {
            return Err(UnitError::ComplexOffset {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        Ok(Complex::new(c.apply(self.re), c.apply(self.im)))
    }
    fn powi(self, n: i32) -> Self {
        Complex::powi(&self, n)
    }
    fn root(self, n: i32) -> Self {
        self.powf(1.0 / f64::from(n))
    }
}

/// Something that names a unit: a [`Unit`], a [`UnitId`], or a string to parse.
pub trait IntoUnit: Sized {
    /// The unit.
    ///
    /// # Errors
    /// [`UnitError`] if a string does not parse.
    fn into_unit(self) -> Result<Unit, UnitError>;

    /// The unit, interned.
    ///
    /// # Errors
    /// As for [`into_unit`](Self::into_unit), and [`UnitError::InternFull`].
    fn into_unit_id(self) -> Result<UnitId, UnitError> {
        UnitId::intern(self.into_unit()?)
    }
}

impl IntoUnit for UnitId {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(self.unit().clone())
    }
    fn into_unit_id(self) -> Result<UnitId, UnitError> {
        Ok(self)
    }
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
    fn into_unit_id(self) -> Result<UnitId, UnitError> {
        UnitId::parse(self)
    }
}

impl IntoUnit for &String {
    fn into_unit(self) -> Result<Unit, UnitError> {
        self.parse()
    }
    fn into_unit_id(self) -> Result<UnitId, UnitError> {
        UnitId::parse(self)
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
/// A quantity is `Copy`: its unit is a [`UnitId`], interned when the quantity is made (from a
/// string, parsed once per spelling).
///
/// The number is an `f64` (the default) or, for `Quantity<K, Complex<f64>>`, a complex number:
/// conversion scales both parts (a unit with an offset, such as °C, is refused), and complex
/// quantities have no order (no `<`). A real quantity becomes a complex one with `into()`.
///
/// ```
/// use annoray_units::Quantity;
/// use annoray_units::dims::{Area, Length, Voltage};
/// use num_complex::Complex;
///
/// let w = Quantity::<Length>::new(3.0, "m")?;
/// let h = Quantity::<Length>::new(40.0, "cm")?;
/// let a: Quantity<Area> = (&w * &h.to("m")?)?;
/// assert_eq!(a.to_string(), "1.2000000000000002 m2");
/// assert!(Quantity::<Length>::new(1.0, "s").is_err());   // not a length
/// assert!((&w + &h).is_err());                            // m + cm: convert first
///
/// let v = Quantity::<Voltage, _>::new(Complex::new(3.0, 4.0), "V")?;
/// assert_eq!(v.to("mV")?.value(), Complex::new(3000.0, 4000.0));
/// assert_eq!(v.norm().to_string(), "5 V");
/// # Ok::<(), annoray_units::UnitError>(())
/// ```
///
/// Complex quantities are not ordered, so comparing them does not compile:
///
/// ```compile_fail
/// use annoray_units::{Dyn, Quantity};
/// use num_complex::Complex;
///
/// let a = Quantity::<Dyn, _>::new(Complex::new(1.0, 2.0), "V").unwrap();
/// let _ = a < a.clone();
/// ```
pub struct Quantity<K: Kind = Dyn, T: Scalar = f64> {
    value: T,
    unit: UnitId,
    kind: PhantomData<K>,
}

impl<K: Kind, T: Scalar> Clone for Quantity<K, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: Kind, T: Scalar> Copy for Quantity<K, T> {}

impl<K: Kind, T: Scalar> Quantity<K, T> {
    const fn raw(value: T, unit: UnitId) -> Self {
        Self {
            value,
            unit,
            kind: PhantomData,
        }
    }

    /// `value` in `unit` (a [`Unit`], a [`UnitId`] or a string).
    ///
    /// # Errors
    /// [`UnitError`] if the unit does not parse, or ([`UnitError::Incompatible`]) is not of
    /// this quantity's dimension; [`UnitError::InternFull`] for a new unit when the table of
    /// interned units is full.
    pub fn new(value: T, unit: impl IntoUnit) -> Result<Self, UnitError> {
        let unit = unit.into_unit_id()?;
        K::check(&unit)?;
        Ok(Self::raw(value, unit))
    }

    /// The number, in [`unit`](Self::unit).
    #[must_use]
    pub const fn value(&self) -> T {
        self.value
    }

    /// The unit.
    #[must_use]
    pub fn unit(&self) -> &'static Unit {
        self.unit.unit()
    }

    /// The unit's handle.
    #[must_use]
    pub const fn unit_id(&self) -> UnitId {
        self.unit
    }

    /// The value and the unit.
    #[must_use]
    pub fn into_parts(self) -> (T, Unit) {
        (self.value, self.unit.into())
    }

    /// The same quantity in another unit of its dimension.
    ///
    /// # Errors
    /// [`UnitError`] if the unit does not parse or measures something else;
    /// [`UnitError::ComplexOffset`] for a complex quantity and a unit with an offset;
    /// [`UnitError::InternFull`].
    pub fn to(&self, unit: impl IntoUnit) -> Result<Self, UnitError> {
        let unit = unit.into_unit_id()?;
        let c = self.unit.converter_to(&unit)?;
        Ok(Self::raw(self.value.convert(&c, self.unit(), &unit)?, unit))
    }

    /// As a quantity whose dimension is checked at run time.
    #[must_use]
    pub fn dynamic(self) -> Quantity<Dyn, T> {
        Quantity::raw(self.value, self.unit)
    }

    /// As a quantity of dimension `D`, if its unit is one.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if the unit has another dimension.
    pub fn cast<D: Kind>(self) -> Result<Quantity<D, T>, UnitError> {
        D::check(&self.unit)?;
        Ok(Quantity::raw(self.value, self.unit))
    }

    /// `self^n` (dimension checked at run time; for a typed result, multiply).
    ///
    /// # Errors
    /// [`UnitError`] for non-linear units (absolute temperatures, …);
    /// [`UnitError::InternFull`].
    pub fn powi(&self, n: i32) -> Result<Quantity<Dyn, T>, UnitError> {
        let unit = self.keep(self.unit.powi(n)?, &[])?;
        Ok(Quantity::raw(self.value.powi(n), unit))
    }

    /// The `n`-th root, the principal one of a complex number (dimension checked at run
    /// time).
    ///
    /// # Errors
    /// [`UnitError::InexactRoot`] if the unit has no exact root; [`UnitError::InternFull`].
    pub fn root(&self, n: i32) -> Result<Quantity<Dyn, T>, UnitError> {
        let unit = self.keep(self.unit.root(n)?, &[])?;
        Ok(Quantity::raw(self.value.root(n), unit))
    }

    /// The value in the dimensionless unit `1` (`5 %` is `0.05`).
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] unless the quantity is dimensionless.
    pub fn as_number(&self) -> Result<T, UnitError> {
        let one = Unit::one();
        self.value
            .convert(&self.unit.converter_to(&one)?, self.unit(), &one)
    }

    /// `self + rhs` or `self - rhs` under `policy` (`+`/`-` use the strict policy: the units
    /// must match).
    ///
    /// # Errors
    /// Unit errors: other dimensions, other scales (strict policy), absolute temperatures;
    /// [`UnitError::InternFull`].
    pub fn add_with(&self, rhs: &Self, sub: bool, policy: UnitPolicy) -> Result<Self, UnitError> {
        let plan = if sub {
            self.unit.plan_sub(&rhs.unit, policy)?
        } else {
            self.unit.plan_add(&rhs.unit, policy)?
        };
        let a = match &plan.lhs {
            Some(c) => self.value.convert(c, self.unit(), &plan.result)?,
            None => self.value,
        };
        let b = match &plan.rhs {
            Some(c) => rhs.value.convert(c, rhs.unit(), &plan.result)?,
            None => rhs.value,
        };
        let unit = self.keep(plan.result, &[rhs.unit])?;
        Ok(Self::raw(if sub { a.minus(b) } else { a.plus(b) }, unit))
    }

    /// The same quantity, as kind `B` (only called where `K` and `B` are the same type).
    fn same<B: Kind>(self) -> Quantity<B, T> {
        Quantity::raw(self.value, self.unit)
    }

    /// The handle of a result unit: this quantity's or one of `others` when spelled the same
    /// (no lookup in the table), otherwise interned.
    fn keep(&self, unit: Unit, others: &[UnitId]) -> Result<UnitId, UnitError> {
        let same = |id: &UnitId| **id == unit && id.label() == unit.label();
        match std::iter::once(&self.unit)
            .chain(others)
            .find(|id| same(id))
        {
            Some(id) => Ok(*id),
            None => UnitId::intern(unit),
        }
    }

    fn product(&self, value: T, unit: UnitId, div: bool) -> Result<(T, UnitId), UnitError> {
        let (v, u) = if div {
            (self.value.over(value), self.unit.div(&unit)?)
        } else {
            (self.value.times(value), self.unit.mul(&unit)?)
        };
        // A plain number keeps the unit as written (`2 × 3 V` is `6 V`, not `6 kg m2 s-3 A-1`),
        // as for arrays; the product above still refuses absolute temperatures.
        let one = Unit::one();
        let u = if *unit == one {
            self.unit
        } else if *self.unit == one && !div {
            unit
        } else {
            self.keep(u, &[unit])?
        };
        Ok((v, u))
    }

    /// Both values in one unit, converting compatible ones; `None` if incompatible.
    fn common(&self, other: &Self) -> Option<(T, T)> {
        let plan = self
            .unit
            .plan_cmp(&other.unit, UnitPolicy::AutoRescale)
            .ok()?;
        let a = match &plan.lhs {
            Some(c) => self.value.convert(c, self.unit(), &plan.result).ok()?,
            None => self.value,
        };
        let b = match &plan.rhs {
            Some(c) => other.value.convert(c, other.unit(), &plan.result).ok()?,
            None => other.value,
        };
        Some((a, b))
    }
}

impl<K: Kind> Quantity<K> {
    /// Compare with another quantity of the same kind, converting compatible units.
    /// `None` if the units are incompatible (possible for `Dyn`) or a value is NaN.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Option<Ordering> {
        let (a, b) = self.common(other)?;
        a.partial_cmp(&b)
    }
}

impl<K: Kind> Quantity<K, Complex<f64>> {
    /// The real part, in the same unit.
    #[must_use]
    pub fn re(&self) -> Quantity<K> {
        Quantity::raw(self.value.re, self.unit)
    }

    /// The imaginary part, in the same unit.
    #[must_use]
    pub fn im(&self) -> Quantity<K> {
        Quantity::raw(self.value.im, self.unit)
    }

    /// The complex conjugate.
    #[must_use]
    pub fn conj(&self) -> Self {
        Self::raw(self.value.conj(), self.unit)
    }

    /// The magnitude (absolute value), in the same unit.
    #[must_use]
    pub fn norm(&self) -> Quantity<K> {
        Quantity::raw(self.value.norm(), self.unit)
    }

    /// The argument (phase), in radians.
    ///
    /// # Errors
    /// Never in practice: `rad` is a known unit.
    pub fn arg(&self) -> Result<Quantity<Dyn>, UnitError> {
        Quantity::new(self.value.arg(), "rad")
    }
}

impl<K: Kind> From<Quantity<K>> for Quantity<K, Complex<f64>> {
    /// A real quantity as a complex one with no imaginary part.
    fn from(q: Quantity<K>) -> Self {
        Self::raw(Complex::new(q.value, 0.0), q.unit)
    }
}

impl<K: Kind, T: Scalar> PartialEq for Quantity<K, T> {
    /// Equal after converting compatible units (`1 km == 1000 m`).
    fn eq(&self, other: &Self) -> bool {
        self.common(other).is_some_and(|(a, b)| a == b)
    }
}

impl<K: Kind> PartialOrd for Quantity<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare(other)
    }
}

impl<K: Kind, T: Scalar> fmt::Display for Quantity<K, T> {
    /// `2.5 km`, `3+4i V`; the format spec applies to the number (`{:.2}` gives `2.50 km`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)?;
        write!(f, " {}", self.unit)
    }
}

impl<K: Kind, T: Scalar> fmt::Debug for Quantity<K, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({:?}, {:?})", self.value, self.unit.to_string())
    }
}

impl<K: Kind, T: Scalar> Neg for Quantity<K, T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::raw(self.value.negated(), self.unit)
    }
}

impl<K: Kind, T: Scalar> Neg for &Quantity<K, T> {
    type Output = Quantity<K, T>;
    fn neg(self) -> Quantity<K, T> {
        Quantity::raw(self.value.negated(), self.unit)
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

impl<A: Kind + SameDimension<B>, B: Kind, T: Scalar> Add<&Quantity<B, T>> for &Quantity<A, T> {
    type Output = Result<Quantity<A, T>, UnitError>;
    fn add(self, rhs: &Quantity<B, T>) -> Self::Output {
        self.add_with(&rhs.same(), false, UnitPolicy::Strict)
    }
}

impl<A: Kind + SameDimension<B>, B: Kind, T: Scalar> Sub<&Quantity<B, T>> for &Quantity<A, T> {
    type Output = Result<Quantity<A, T>, UnitError>;
    fn sub(self, rhs: &Quantity<B, T>) -> Self::Output {
        self.add_with(&rhs.same(), true, UnitPolicy::Strict)
    }
}

forward!(impl[A: Kind + SameDimension<B>, B: Kind, T: Scalar] Add add for Quantity<A, T>, Quantity<B, T> => Result<Quantity<A, T>, UnitError>);
forward!(impl[A: Kind + SameDimension<B>, B: Kind, T: Scalar] Sub sub for Quantity<A, T>, Quantity<B, T> => Result<Quantity<A, T>, UnitError>);

// Typed products: the dimension of the result is computed by the compiler.
impl<A: Dimensional + Mul<B>, B: Dimensional, T: Scalar> Mul<&Quantity<B, T>> for &Quantity<A, T>
where
    Prod<A, B>: Dimensional,
{
    type Output = Result<Quantity<Prod<A, B>, T>, UnitError>;
    fn mul(self, rhs: &Quantity<B, T>) -> Self::Output {
        let (v, u) = self.product(rhs.value, rhs.unit, false)?;
        Ok(Quantity::raw(v, u))
    }
}

impl<A: Dimensional + Div<B>, B: Dimensional, T: Scalar> Div<&Quantity<B, T>> for &Quantity<A, T>
where
    Quot<A, B>: Dimensional,
{
    type Output = Result<Quantity<Quot<A, B>, T>, UnitError>;
    fn div(self, rhs: &Quantity<B, T>) -> Self::Output {
        let (v, u) = self.product(rhs.value, rhs.unit, true)?;
        Ok(Quantity::raw(v, u))
    }
}

forward!(impl[A: Dimensional + Mul<B>, B: Dimensional, T: Scalar] Mul mul for Quantity<A, T>, Quantity<B, T>
    => Result<Quantity<Prod<A, B>, T>, UnitError> where Prod<A, B>: Dimensional);
forward!(impl[A: Dimensional + Div<B>, B: Dimensional, T: Scalar] Div div for Quantity<A, T>, Quantity<B, T>
    => Result<Quantity<Quot<A, B>, T>, UnitError> where Quot<A, B>: Dimensional);

// Dynamic products (either side `Dyn`): dimension checked at run time.
macro_rules! dyn_products {
    ($(impl[$($g:tt)*] $l:ty, $r:ty;)*) => {$(
        impl<'a, 'b, T: Scalar, $($g)*> Mul<&'b Quantity<$r, T>> for &'a Quantity<$l, T> {
            type Output = Result<Quantity<Dyn, T>, UnitError>;
            fn mul(self, rhs: &'b Quantity<$r, T>) -> Self::Output {
                let (v, u) = self.product(rhs.value, rhs.unit, false)?;
                Ok(Quantity::raw(v, u))
            }
        }
        impl<'a, 'b, T: Scalar, $($g)*> Div<&'b Quantity<$r, T>> for &'a Quantity<$l, T> {
            type Output = Result<Quantity<Dyn, T>, UnitError>;
            fn div(self, rhs: &'b Quantity<$r, T>) -> Self::Output {
                let (v, u) = self.product(rhs.value, rhs.unit, true)?;
                Ok(Quantity::raw(v, u))
            }
        }
        forward!(impl[T: Scalar, $($g)*] Mul mul for Quantity<$l, T>, Quantity<$r, T> => Result<Quantity<Dyn, T>, UnitError>);
        forward!(impl[T: Scalar, $($g)*] Div div for Quantity<$l, T>, Quantity<$r, T> => Result<Quantity<Dyn, T>, UnitError>);
    )*};
}

dyn_products! {
    impl[] Dyn, Dyn;
    impl[D: Dimensional] D, Dyn;
    impl[D: Dimensional] Dyn, D;
}

// Plain numbers are dimensionless: scaling keeps the unit (an error for absolute
// temperatures, as for arrays). Real numbers scale any quantity; complex numbers scale complex
// ones.
macro_rules! scalar_products {
    ($($num:ty => $t:ty),*) => {$(
        impl<K: Kind> Mul<$num> for &Quantity<K, $t> {
            type Output = Result<Quantity<K, $t>, UnitError>;
            fn mul(self, rhs: $num) -> Self::Output {
                let (v, u) = self.product(<$t>::from(rhs), UnitId::one(), false)?;
                Ok(Quantity::raw(v, u))
            }
        }

        impl<K: Kind> Div<$num> for &Quantity<K, $t> {
            type Output = Result<Quantity<K, $t>, UnitError>;
            fn div(self, rhs: $num) -> Self::Output {
                let (v, u) = self.product(<$t>::from(rhs), UnitId::one(), true)?;
                Ok(Quantity::raw(v, u))
            }
        }

        impl<K: Kind> Mul<$num> for Quantity<K, $t> {
            type Output = Result<Self, UnitError>;
            fn mul(self, rhs: $num) -> Self::Output {
                Mul::mul(&self, rhs)
            }
        }

        impl<K: Kind> Div<$num> for Quantity<K, $t> {
            type Output = Result<Self, UnitError>;
            fn div(self, rhs: $num) -> Self::Output {
                Div::div(&self, rhs)
            }
        }

        impl<K: Kind> Mul<Quantity<K, $t>> for $num {
            type Output = Result<Quantity<K, $t>, UnitError>;
            fn mul(self, rhs: Quantity<K, $t>) -> Self::Output {
                Mul::mul(&rhs, self)
            }
        }

        impl<K: Kind> Mul<&Quantity<K, $t>> for $num {
            type Output = Result<Quantity<K, $t>, UnitError>;
            fn mul(self, rhs: &Quantity<K, $t>) -> Self::Output {
                Mul::mul(rhs, self)
            }
        }
    )*};
}

scalar_products!(f64 => f64, f64 => Complex<f64>, Complex<f64> => Complex<f64>);

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;
    use crate::dims::{Length, Voltage};

    type C = Complex<f64>;

    fn v(re: f64, im: f64) -> Quantity<Voltage, C> {
        Quantity::new(Complex::new(re, im), "V").unwrap()
    }

    #[test]
    fn complex_conversion_scales_both_parts() {
        let q = v(1.5, -2.0);
        assert_eq!(q.to("mV").unwrap().value(), Complex::new(1500.0, -2000.0));
        let c = Quantity::<Dyn, C>::new(Complex::new(1.0, 1.0), "degC").unwrap();
        assert!(matches!(c.to("K"), Err(UnitError::ComplexOffset { .. })));
        // A real quantity converts across an offset as before.
        assert_eq!(
            Quantity::<Dyn>::new(1.0, "degC")
                .unwrap()
                .to("K")
                .unwrap()
                .value(),
            274.15
        );
    }

    #[test]
    fn complex_arithmetic_keeps_unit_rules() {
        let a = v(1.0, 2.0);
        let b = v(3.0, -1.0);
        assert_eq!((a + b).unwrap().value(), Complex::new(4.0, 1.0));
        assert_eq!((a - b).unwrap().value(), Complex::new(-2.0, 3.0));
        let mv = Quantity::<Voltage, C>::new(Complex::new(1.0, 0.0), "mV").unwrap();
        assert!((a + mv).is_err()); // V + mV: convert first, as for real quantities
        let sum = a.add_with(&mv, false, UnitPolicy::AutoRescale).unwrap();
        assert_eq!(sum.value(), Complex::new(1.001, 2.0));
        // Products: a typed dimension, a dynamic one, and plain numbers.
        let p = (a * b).unwrap();
        assert_eq!(p.value(), Complex::new(5.0, 5.0));
        assert_eq!(p.unit().to_string(), "V2");
        let len = Quantity::<Length, C>::new(Complex::new(2.0, 0.0), "m").unwrap();
        let d = (a.dynamic() / len).unwrap();
        assert_eq!(d.value(), Complex::new(0.5, 1.0));
        assert_eq!((&a * 2.0).unwrap().value(), Complex::new(2.0, 4.0));
        assert_eq!((&a * 2.0).unwrap().unit().to_string(), "V"); // the spelling kept
        assert_eq!(
            (a * Complex::new(0.0, 1.0)).unwrap().value(),
            Complex::new(-2.0, 1.0)
        );
        assert_eq!((2.0 * &a).unwrap().value(), Complex::new(2.0, 4.0));
        let two = Quantity::<Dyn, C>::new(Complex::new(2.0, 0.0), "1").unwrap();
        let twice = (two * a.dynamic()).unwrap();
        assert_eq!(twice.unit().to_string(), "V"); // a number on the left keeps it too
        assert_eq!((-&a).value(), Complex::new(-1.0, -2.0));
        assert_eq!(a.powi(2).unwrap().value(), Complex::new(-3.0, 4.0));
        let r = v(-4.0, 0.0).root(2);
        assert!(r.is_err()); // V has no exact square root
        let r = Quantity::<Dyn, C>::new(Complex::new(-4.0, 0.0), "m2")
            .unwrap()
            .root(2)
            .unwrap();
        assert!((r.value() - Complex::new(0.0, 2.0)).norm() < 1e-12);
        assert_eq!(r.unit().to_string(), "m");
    }

    #[test]
    fn parts_and_equality() {
        let a = v(3.0, 4.0);
        assert_eq!(a.re().to_string(), "3 V");
        assert_eq!(a.im().to_string(), "4 V");
        assert_eq!(a.norm().value(), 5.0);
        assert_eq!(a.conj().value(), Complex::new(3.0, -4.0));
        assert!((a.arg().unwrap().value() - 4.0_f64.atan2(3.0)).abs() < 1e-15);
        assert_eq!(a.arg().unwrap().unit().to_string(), "rad");
        assert_eq!(a.to_string(), "3+4i V");
        assert_eq!(format!("{a:.1}"), "3.0+4.0i V");
        // Equal after converting compatible units; never equal across dimensions.
        assert_eq!(a, a.to("mV").unwrap());
        assert_ne!(a, v(3.0, -4.0));
        let real: Quantity<Voltage, C> = Quantity::<Voltage>::new(2.0, "V").unwrap().into();
        assert_eq!(real.value(), Complex::new(2.0, 0.0));
        let one = Quantity::<Dyn, C>::new(Complex::new(50.0, 25.0), "%").unwrap();
        assert_eq!(one.as_number().unwrap(), Complex::new(0.5, 0.25));
    }
}
