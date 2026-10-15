//! Dimensions as types, for [`Quantity`](crate::Quantity): checked by the compiler.
//!
//! A dimension is the exponents of the eight base quantities (the seven SI ones and angle),
//! held at the type level: [`Dim<M, L, T, I, Θ, N, J, A>`](Dim). Multiplying, dividing and
//! raising quantities computes the result's dimension as a type, so `Length / Time` *is*
//! [`Velocity`], whoever writes it and whatever they call it. Named dimensions are type
//! aliases, declared with [`dimension!`](crate::dimension): compose your own from the ones
//! here (or from each other), in any crate.
//!
//! ```
//! use annoray_units::Quantity;
//! use annoray_units::dims::{Length, Time, Velocity};
//!
//! annoray_units::dimension! {
//!     /// Rate of change of acceleration.
//!     pub Jerk = Length / Time^3;
//! }
//!
//! let d = Quantity::<Length>::new(2.5, "km")?;
//! let t = Quantity::<Time>::new(30.0, "min")?;
//! let v: Quantity<Velocity> = (&d / &t)?;
//! let s = t.to("s")?;
//! // A velocity over a time, twice: a jerk. (Three times would not compile.)
//! let j: Quantity<Jerk> = (&(&v.to("m/s")? / &s)? / &s)?;
//! assert!((j.value() - 2500.0 / 1800.0 / 1800.0 / 1800.0).abs() < 1e-15);
//! assert_eq!(j.unit().to_string(), "m s-3");
//! // `d + t` does not compile: lengths and times do not add.
//! # Ok::<(), annoray_units::UnitError>(())
//! ```
//!
//! Units still matter at run time: `Quantity<Length>` holds `km` or `m`, and mixing
//! scales or using absolute temperatures where they make no sense is a [`UnitError`]
//! (as for arrays).
//!
//! [`UnitError`]: crate::UnitError

use std::marker::PhantomData;
use std::ops::{Div, Mul};

use typenum::{Integer, N1, P1, Z0};

use crate::Dimension;

#[doc(hidden)]
pub use typenum as __typenum;

/// A dimension as a type: the exponents of mass, length, time, current, temperature, amount
/// of substance, luminous intensity and angle (type-level integers). Use the aliases
/// ([`Length`], [`Velocity`], …) or [`dimension!`](crate::dimension) rather than writing
/// these out.
pub struct Dim<M, L, T, I, Th, N, J, A>(Exponents<M, L, T, I, Th, N, J, A>);

/// The phantom of [`Dim`]: a function pointer, so `Dim` is `Copy`, `Send` and `Sync` whatever
/// the exponent types.
type Exponents<M, L, T, I, Th, N, J, A> = PhantomData<fn() -> (M, L, T, I, Th, N, J, A)>;

impl<M, L, T, I, Th, N, J, A> Clone for Dim<M, L, T, I, Th, N, J, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M, L, T, I, Th, N, J, A> Copy for Dim<M, L, T, I, Th, N, J, A> {}

impl<M, L, T, I, Th, N, J, A> std::fmt::Debug for Dim<M, L, T, I, Th, N, J, A>
where
    Self: Dimensional,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dim({})", Self::DIMENSION)
    }
}

/// A type that stands for a physical dimension (implemented by [`Dim`]).
pub trait Dimensional: Copy + Send + Sync + 'static {
    /// The dimension, as a runtime value.
    const DIMENSION: Dimension;
}

impl<M, L, T, I, Th, N, J, A> Dimensional for Dim<M, L, T, I, Th, N, J, A>
where
    M: Integer,
    L: Integer,
    T: Integer,
    I: Integer,
    Th: Integer,
    N: Integer,
    J: Integer,
    A: Integer,
{
    const DIMENSION: Dimension =
        Dimension::from_exponents([M::I8, L::I8, T::I8, I::I8, Th::I8, N::I8, J::I8, A::I8]);
}

macro_rules! dim_op {
    ($tr:ident $m:ident $int:ident) => {
        impl<M1, L1, T1, I1, Th1, N1_, J1, A1, M2, L2, T2, I2, Th2, N2, J2, A2>
            $tr<Dim<M2, L2, T2, I2, Th2, N2, J2, A2>> for Dim<M1, L1, T1, I1, Th1, N1_, J1, A1>
        where
            M1: std::ops::$int<M2>,
            L1: std::ops::$int<L2>,
            T1: std::ops::$int<T2>,
            I1: std::ops::$int<I2>,
            Th1: std::ops::$int<Th2>,
            N1_: std::ops::$int<N2>,
            J1: std::ops::$int<J2>,
            A1: std::ops::$int<A2>,
        {
            type Output = Dim<
                <M1 as std::ops::$int<M2>>::Output,
                <L1 as std::ops::$int<L2>>::Output,
                <T1 as std::ops::$int<T2>>::Output,
                <I1 as std::ops::$int<I2>>::Output,
                <Th1 as std::ops::$int<Th2>>::Output,
                <N1_ as std::ops::$int<N2>>::Output,
                <J1 as std::ops::$int<J2>>::Output,
                <A1 as std::ops::$int<A2>>::Output,
            >;
            fn $m(self, _: Dim<M2, L2, T2, I2, Th2, N2, J2, A2>) -> Self::Output {
                Dim(PhantomData)
            }
        }
    };
}

// Dimensions multiply by adding exponents and divide by subtracting them.
dim_op!(Mul mul Add);
dim_op!(Div div Sub);

/// A dimension to an integer power `P` (a type-level integer).
pub trait DimPow<P> {
    /// The resulting dimension.
    type Output;
}

impl<M, L, T, I, Th, N, J, A, P> DimPow<P> for Dim<M, L, T, I, Th, N, J, A>
where
    M: Mul<P>,
    L: Mul<P>,
    T: Mul<P>,
    I: Mul<P>,
    Th: Mul<P>,
    N: Mul<P>,
    J: Mul<P>,
    A: Mul<P>,
{
    type Output = Dim<
        <M as Mul<P>>::Output,
        <L as Mul<P>>::Output,
        <T as Mul<P>>::Output,
        <I as Mul<P>>::Output,
        <Th as Mul<P>>::Output,
        <N as Mul<P>>::Output,
        <J as Mul<P>>::Output,
        <A as Mul<P>>::Output,
    >;
}

/// The dimension of a product.
pub type Prod<A, B> = <A as Mul<B>>::Output;
/// The dimension of a quotient.
pub type Quot<A, B> = <A as Div<B>>::Output;
/// A dimension to the power `P` (a type-level integer such as `typenum::P3`).
pub type Pow<A, P> = <A as DimPow<P>>::Output;

/// Declare named dimensions composed of others: `pub Velocity = Length / Time;`,
/// `pub Jerk = Length / Time^3;`, `pub Momentum = Mass * Velocity;`. Each becomes a type
/// alias of the computed [`Dim`], so names that describe the same exponents are the same
/// type (`Speed` and `Velocity` interchange). Powers are integers from -9 to 9.
#[macro_export]
macro_rules! dimension {
    () => {};
    // Exponent on the first factor.
    (@decl [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ident] ; ^ $($rest:tt)*) => {
        $crate::dimension!(@pow [$($meta)*] $vis $name ; $acc ; $($rest)*);
    };
    (@pow [$($meta:tt)*] $vis:vis $name:ident ; $base:ident ; - $n:tt $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ;
            [$crate::dims::Pow<$base, $crate::dimension!(@neg $n)>] ; $($rest)*);
    };
    (@pow [$($meta:tt)*] $vis:vis $name:ident ; $base:ident ; $n:tt $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ;
            [$crate::dims::Pow<$base, $crate::dimension!(@pos $n)>] ; $($rest)*);
    };
    (@decl [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ident] ; $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ; [$acc] ; $($rest)*);
    };
    // End of one declaration.
    (@decl2 [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ty] ; ; $($more:tt)*) => {
        $($meta)*
        $vis type $name = $acc;
        $crate::dimension!($($more)*);
    };
    // Further factors: `* B`, `/ B`, with an optional `^ n`.
    (@decl2 [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ty] ; $op:tt $b:ident ^ - $n:tt $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ;
            [$crate::dimension!(@op $op $acc, $crate::dims::Pow<$b, $crate::dimension!(@neg $n)>)] ; $($rest)*);
    };
    (@decl2 [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ty] ; $op:tt $b:ident ^ $n:tt $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ;
            [$crate::dimension!(@op $op $acc, $crate::dims::Pow<$b, $crate::dimension!(@pos $n)>)] ; $($rest)*);
    };
    (@decl2 [$($meta:tt)*] $vis:vis $name:ident ; [$acc:ty] ; $op:tt $b:ident $($rest:tt)*) => {
        $crate::dimension!(@decl2 [$($meta)*] $vis $name ;
            [$crate::dimension!(@op $op $acc, $b)] ; $($rest)*);
    };
    (@op * $a:ty, $b:ty) => { $crate::dims::Prod<$a, $b> };
    (@op / $a:ty, $b:ty) => { $crate::dims::Quot<$a, $b> };
    (@pos 0) => { $crate::dims::__typenum::Z0 };
    (@pos 1) => { $crate::dims::__typenum::P1 };
    (@pos 2) => { $crate::dims::__typenum::P2 };
    (@pos 3) => { $crate::dims::__typenum::P3 };
    (@pos 4) => { $crate::dims::__typenum::P4 };
    (@pos 5) => { $crate::dims::__typenum::P5 };
    (@pos 6) => { $crate::dims::__typenum::P6 };
    (@pos 7) => { $crate::dims::__typenum::P7 };
    (@pos 8) => { $crate::dims::__typenum::P8 };
    (@pos 9) => { $crate::dims::__typenum::P9 };
    (@neg 1) => { $crate::dims::__typenum::N1 };
    (@neg 2) => { $crate::dims::__typenum::N2 };
    (@neg 3) => { $crate::dims::__typenum::N3 };
    (@neg 4) => { $crate::dims::__typenum::N4 };
    (@neg 5) => { $crate::dims::__typenum::N5 };
    (@neg 6) => { $crate::dims::__typenum::N6 };
    (@neg 7) => { $crate::dims::__typenum::N7 };
    (@neg 8) => { $crate::dims::__typenum::N8 };
    (@neg 9) => { $crate::dims::__typenum::N9 };
    ($(#[$meta:meta])* $vis:vis $name:ident = $first:ident $($rest:tt)*) => {
        $crate::dimension!(@decl [$(#[$meta])*] $vis $name ; [$first] ; $($rest)*);
    };
}

// ----- the base quantities -----

/// No dimension (pure numbers, ratios).
pub type Dimensionless = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Mass (kg).
pub type Mass = Dim<P1, Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Length (m).
pub type Length = Dim<Z0, P1, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Time (s).
pub type Time = Dim<Z0, Z0, P1, Z0, Z0, Z0, Z0, Z0>;
/// Electric current (A).
pub type Current = Dim<Z0, Z0, Z0, P1, Z0, Z0, Z0, Z0>;
/// Thermodynamic temperature (K).
pub type Temperature = Dim<Z0, Z0, Z0, Z0, P1, Z0, Z0, Z0>;
/// Amount of substance (mol).
pub type Amount = Dim<Z0, Z0, Z0, Z0, Z0, P1, Z0, Z0>;
/// Luminous intensity (cd).
pub type Luminosity = Dim<Z0, Z0, Z0, Z0, Z0, Z0, P1, Z0>;
/// Plane angle (rad): a dimension of its own here, as in whippyunits.
pub type Angle = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0, P1>;

/// Frequency (s-1).
pub type Frequency = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0, Z0>;

// ----- common derived dimensions, declared as users declare theirs -----

crate::dimension! {
    /// Area (m2).
    pub Area = Length^2;
    /// Volume (m3).
    pub Volume = Length^3;
    /// Velocity (m s-1).
    pub Velocity = Length / Time;
    /// Acceleration (m s-2).
    pub Acceleration = Velocity / Time;
    /// Force (N = kg m s-2).
    pub Force = Mass * Acceleration;
    /// Momentum (kg m s-1).
    pub Momentum = Mass * Velocity;
    /// Energy, work and heat (J = N m). Torque has the same dimension.
    pub Energy = Force * Length;
    /// Power (W = J s-1).
    pub Power = Energy / Time;
    /// Pressure and stress (Pa = N m-2).
    pub Pressure = Force / Area;
    /// Density (kg m-3).
    pub Density = Mass / Volume;
    /// Electric charge (C = A s).
    pub Charge = Current * Time;
    /// Voltage (V = W A-1).
    pub Voltage = Power / Current;
    /// Resistance (Ω = V A-1).
    pub Resistance = Voltage / Current;
    /// Angular velocity (rad s-1).
    pub AngularVelocity = Angle / Time;
    /// Concentration (mol m-3).
    pub Concentration = Amount / Volume;
}
