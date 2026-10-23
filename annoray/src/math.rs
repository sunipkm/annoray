//! Elementwise mathematical functions with units (numpy's unary ufuncs), and powers.
//!
//! - Sign and magnitude (`Neg`, `Abs`) keep a ratio-scale or logarithmic unit and refuse
//!   interval and epoch units (−5 °C is not the negation of 5 °C); rounding (`Floor`, `Ceil`,
//!   `Trunc`, `Rint`) keeps any unit.
//! - Roots and powers transform the unit (`Sqrt` of m² is m; an inexact root, such as the
//!   square root of m, is an error); `Reciprocal` inverts it.
//! - Exponentials, logarithms and hyperbolic functions need a dimensionless argument (scaled
//!   ones such as percent are converted to plain numbers first) and give a dimensionless
//!   result. `Sin`, `Cos`, `Tan` take angles (converted to radians) or plain numbers; the
//!   inverse functions take plain numbers and give radians. `Sign` gives plain numbers.
//! - Types as numpy: `f32` stays `f32`; integers stay integers for `Neg`, `Abs`, `Square`,
//!   `Conj`, `Real` and `Imag` (checked; overflow is an error), and become `f64` otherwise;
//!   booleans and non-numeric data are refused. NaN propagates.
//! - Complex numbers: `Conj`, `Real`, `Imag` and `Angle` (radians) take any numeric data
//!   (for real data the conjugate and real part are the data, the imaginary part zero, the
//!   angle 0 or π). `Abs`, `Real`, `Imag` and `Angle` of complex data are real (`float32` for
//!   `complex64`); `Neg`, `Square`, `Reciprocal`, `Sqrt`, the exponential and logarithms, and
//!   the trigonometric and hyperbolic functions and their inverses stay complex (principal
//!   branches). Rounding, `Sign`, `Cbrt`, `Exp2`, `Expm1` and `Log1p` are refused for complex
//!   data, as are units with an offset (°C).
//! - Lazy data stays lazy (applied block by block when read).

use annoray_units::{Converter, Unit, UnitError, UnitKind};
use ndarray::ArrayD;
use num_complex::Complex;

use crate::{
    CoreError, DType, DataArray, DataArrayBase, Repr, Result, TypedArray, Variable, VariableBase,
};

/// An elementwise function (numpy's ufunc of the same name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum MathFn {
    Neg,
    Abs,
    Sqrt,
    Cbrt,
    Square,
    Reciprocal,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Floor,
    Ceil,
    Trunc,
    Rint,
    Sign,
    /// The complex conjugate.
    Conj,
    /// The real part.
    Real,
    /// The imaginary part.
    Imag,
    /// The argument (phase angle) of a complex number, in radians.
    Angle,
}

impl MathFn {
    /// The function for a numpy ufunc name (`"sqrt"`, `"negative"`, `"absolute"`, …).
    #[must_use]
    pub fn from_numpy(name: &str) -> Option<Self> {
        Some(match name {
            "negative" => Self::Neg,
            "absolute" | "fabs" => Self::Abs,
            "sqrt" => Self::Sqrt,
            "cbrt" => Self::Cbrt,
            "square" => Self::Square,
            "reciprocal" => Self::Reciprocal,
            "exp" => Self::Exp,
            "exp2" => Self::Exp2,
            "expm1" => Self::Expm1,
            "log" => Self::Log,
            "log2" => Self::Log2,
            "log10" => Self::Log10,
            "log1p" => Self::Log1p,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "arcsin" => Self::Arcsin,
            "arccos" => Self::Arccos,
            "arctan" => Self::Arctan,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "arcsinh" => Self::Arcsinh,
            "arccosh" => Self::Arccosh,
            "arctanh" => Self::Arctanh,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "trunc" => Self::Trunc,
            "rint" => Self::Rint,
            "sign" => Self::Sign,
            "conjugate" | "conj" => Self::Conj,
            "real" => Self::Real,
            "imag" => Self::Imag,
            "angle" => Self::Angle,
            _ => return None,
        })
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Neg => "negative",
            Self::Abs => "absolute",
            Self::Sqrt => "sqrt",
            Self::Cbrt => "cbrt",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Expm1 => "expm1",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Arcsin => "arcsin",
            Self::Arccos => "arccos",
            Self::Arctan => "arctan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Arcsinh => "arcsinh",
            Self::Arccosh => "arccosh",
            Self::Arctanh => "arctanh",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Rint => "rint",
            Self::Sign => "sign",
            Self::Conj => "conjugate",
            Self::Real => "real",
            Self::Imag => "imag",
            Self::Angle => "angle",
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Neg => -x,
            Self::Abs => x.abs(),
            Self::Sqrt => x.sqrt(),
            Self::Cbrt => x.cbrt(),
            Self::Square => x * x,
            Self::Reciprocal => 1.0 / x,
            Self::Exp => x.exp(),
            Self::Exp2 => x.exp2(),
            Self::Expm1 => x.exp_m1(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Log10 => x.log10(),
            Self::Log1p => x.ln_1p(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Arcsin => x.asin(),
            Self::Arccos => x.acos(),
            Self::Arctan => x.atan(),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Arcsinh => x.asinh(),
            Self::Arccosh => x.acosh(),
            Self::Arctanh => x.atanh(),
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Trunc => x.trunc(),
            Self::Rint => x.round_ties_even(),
            Self::Conj | Self::Real => x,
            Self::Imag => 0.0,
            // 0 for positive numbers, π for negative ones (and for -0.0), as numpy's `angle`.
            Self::Angle => 0f64.atan2(x),
            Self::Sign => {
                if x.is_nan() {
                    f64::NAN
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn apply_f32(self, x: f32) -> f32 {
        match self {
            Self::Neg => -x,
            Self::Abs => x.abs(),
            Self::Sqrt => x.sqrt(),
            Self::Square => x * x,
            Self::Reciprocal => 1.0 / x,
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Trunc => x.trunc(),
            Self::Rint => x.round_ties_even(),
            #[allow(clippy::cast_possible_truncation)] // f32 data stays f32
            other => other.apply(f64::from(x)) as f32,
        }
    }
}

/// The input conversion and result unit of `f` for data in `unit`.
fn plan(f: MathFn, unit: Option<&Unit>) -> Result<(Option<Converter>, Option<Unit>)> {
    let refuse = |u: &Unit| -> CoreError {
        UnitError::UnsupportedOp {
            op: f.name(),
            unit: u.to_string(),
        }
        .into()
    };
    let Some(u) = unit else {
        return Ok((None, None));
    };
    let one = Unit::one();
    let plain = |u: &Unit| -> Result<Converter> { u.converter_to(&one).map_err(|_| refuse(u)) };
    Ok(match f {
        MathFn::Neg | MathFn::Abs => match u.kind() {
            UnitKind::Linear | UnitKind::Log(_) => (None, Some(u.clone())),
            _ => return Err(refuse(u)),
        },
        MathFn::Floor
        | MathFn::Ceil
        | MathFn::Trunc
        | MathFn::Rint
        | MathFn::Conj
        | MathFn::Real
        | MathFn::Imag => (None, Some(u.clone())),
        MathFn::Angle => (None, Some("rad".parse()?)),
        MathFn::Sqrt => (None, Some(u.root(2)?)),
        MathFn::Cbrt => (None, Some(u.root(3)?)),
        MathFn::Square => (None, Some(u.powi(2)?)),
        MathFn::Reciprocal => (None, Some(u.powi(-1)?)),
        MathFn::Sin | MathFn::Cos | MathFn::Tan => {
            let rad: Unit = "rad".parse()?;
            match u.converter_to(&rad) {
                Ok(c) => (Some(c), None),
                Err(_) => (Some(plain(u)?), None),
            }
        }
        MathFn::Arcsin | MathFn::Arccos | MathFn::Arctan => (Some(plain(u)?), Some("rad".parse()?)),
        MathFn::Sign => (None, None),
        _ => (Some(plain(u)?), None),
    })
}

fn compute(
    f: MathFn,
    t: &TypedArray<crate::Owned>,
    conv: Option<&Converter>,
) -> Result<TypedArray<crate::Owned>> {
    let unsupported = |d: DType| CoreError::UnsupportedDType {
        op: f.name(),
        dtype: d,
    };
    let keeps_int = matches!(
        f,
        MathFn::Neg | MathFn::Abs | MathFn::Square | MathFn::Conj | MathFn::Real | MathFn::Imag
    );
    macro_rules! int {
        ($a:expr, $t:ty, $var:ident) => {{
            if keeps_int && conv.is_none() {
                let mut out = $a.clone();
                for x in &mut out {
                    let v: Option<$t> = match f {
                        MathFn::Neg => (0 as $t).checked_sub(*x),
                        MathFn::Abs => int_abs(*x),
                        MathFn::Conj | MathFn::Real => Some(*x),
                        MathFn::Imag => Some(0 as $t),
                        _ => x.checked_mul(*x),
                    };
                    *x = v.ok_or(CoreError::ArithmeticOverflow("integer math"))?;
                }
                return Ok(TypedArray::$var(out));
            }
        }};
    }
    match t {
        TypedArray::C64(a) => {
            return complex_math(f, a, conv, TypedArray::C64, TypedArray::F32);
        }
        TypedArray::C128(a) => {
            return complex_math(f, a, conv, TypedArray::C128, TypedArray::F64);
        }
        TypedArray::I8(a) => int!(a, i8, I8),
        TypedArray::I16(a) => int!(a, i16, I16),
        TypedArray::I32(a) => int!(a, i32, I32),
        TypedArray::I64(a) => int!(a, i64, I64),
        TypedArray::U8(a) => int!(a, u8, U8),
        TypedArray::U16(a) => int!(a, u16, U16),
        TypedArray::U32(a) => int!(a, u32, U32),
        TypedArray::U64(a) => int!(a, u64, U64),
        _ => {}
    }
    match t {
        TypedArray::F32(a) if conv.is_none_or(Converter::is_identity) => {
            Ok(TypedArray::F32(a.mapv(|x| f.apply_f32(x))))
        }
        TypedArray::Bool(_)
        | TypedArray::Str(_)
        | TypedArray::DateTime(_)
        | TypedArray::CfDate(_)
        | TypedArray::TimeDelta(_) => Err(unsupported(t.dtype())),
        other => {
            let x: ArrayD<f64> = crate::ops::to_f64(&other.view(), conv)?;
            #[allow(clippy::cast_possible_truncation)]
            Ok(if other.dtype() == DType::F32 {
                TypedArray::F32(x.mapv(|v| f.apply(v) as f32))
            } else {
                TypedArray::F64(x.mapv(|v| f.apply(v)))
            })
        }
    }
}

/// `f` of complex data with parts of type `P`: complex results wrapped by `complex`, real
/// ones (`Abs`, `Real`, `Imag`, `Angle`) by `real`.
fn complex_math<P: num_traits_float::Part>(
    f: MathFn,
    a: &ArrayD<Complex<P>>,
    conv: Option<&Converter>,
    complex: fn(ArrayD<Complex<P>>) -> TypedArray<crate::Owned>,
    real: fn(ArrayD<P>) -> TypedArray<crate::Owned>,
) -> Result<TypedArray<crate::Owned>> {
    let unsupported = || CoreError::UnsupportedDType {
        op: f.name(),
        dtype: P::COMPLEX,
    };
    let scaled = complex_scaled(a, conv).map_err(|_| unsupported())?;
    Ok(match f {
        MathFn::Abs => real(scaled.mapv(Complex::norm)),
        MathFn::Real => real(scaled.mapv(|z| z.re)),
        MathFn::Imag => real(scaled.mapv(|z| z.im)),
        MathFn::Angle => real(scaled.mapv(Complex::arg)),
        MathFn::Conj => complex(scaled.mapv(|z| z.conj())),
        MathFn::Neg => complex(scaled.mapv(|z| -z)),
        MathFn::Square => complex(scaled.mapv(|z| z * z)),
        MathFn::Reciprocal => complex(scaled.mapv(|z| z.inv())),
        MathFn::Sqrt => complex(scaled.mapv(Complex::sqrt)),
        MathFn::Exp => complex(scaled.mapv(Complex::exp)),
        MathFn::Log => complex(scaled.mapv(Complex::ln)),
        MathFn::Log2 => complex(scaled.mapv(Complex::log2)),
        MathFn::Log10 => complex(scaled.mapv(Complex::log10)),
        MathFn::Sin => complex(scaled.mapv(Complex::sin)),
        MathFn::Cos => complex(scaled.mapv(Complex::cos)),
        MathFn::Tan => complex(scaled.mapv(Complex::tan)),
        MathFn::Arcsin => complex(scaled.mapv(Complex::asin)),
        MathFn::Arccos => complex(scaled.mapv(Complex::acos)),
        MathFn::Arctan => complex(scaled.mapv(Complex::atan)),
        MathFn::Sinh => complex(scaled.mapv(Complex::sinh)),
        MathFn::Cosh => complex(scaled.mapv(Complex::cosh)),
        MathFn::Tanh => complex(scaled.mapv(Complex::tanh)),
        MathFn::Arcsinh => complex(scaled.mapv(Complex::asinh)),
        MathFn::Arccosh => complex(scaled.mapv(Complex::acosh)),
        MathFn::Arctanh => complex(scaled.mapv(Complex::atanh)),
        MathFn::Cbrt
        | MathFn::Exp2
        | MathFn::Expm1
        | MathFn::Log1p
        | MathFn::Floor
        | MathFn::Ceil
        | MathFn::Trunc
        | MathFn::Rint
        | MathFn::Sign => return Err(unsupported()),
    })
}

/// Complex data in the unit `conv` converts to. Scaling (percent to plain numbers, degrees to
/// radians) applies to both parts; an offset (°C) has no meaning for a complex number.
fn complex_scaled<P: num_traits_float::Part>(
    a: &ArrayD<Complex<P>>,
    conv: Option<&Converter>,
) -> Result<ArrayD<Complex<P>>> {
    Ok(match conv {
        Some(c) if c.offset() != 0.0 => {
            return Err(CoreError::UnsupportedDType {
                op: "conversion to an affine unit",
                dtype: P::COMPLEX,
            });
        }
        Some(c) if !c.is_identity() => {
            a.mapv(|z| Complex::new(P::scaled(z.re, c), P::scaled(z.im, c)))
        }
        _ => a.clone(),
    })
}

/// The part types of complex numbers.
mod num_traits_float {
    use annoray_units::Converter;

    use crate::DType;

    /// `f32` or `f64`, the parts of `complex64` and `complex128`.
    pub(crate) trait Part:
        num_traits::Float + num_traits::FloatConst + Send + Sync + 'static
    {
        /// The complex element type with these parts.
        const COMPLEX: DType;
        /// `c` applied to `x` (a scaling: the caller has refused offsets).
        fn scaled(x: Self, c: &Converter) -> Self;
    }

    impl Part for f32 {
        const COMPLEX: DType = DType::C64;
        #[allow(clippy::cast_possible_truncation)] // complex64 data stays complex64
        fn scaled(x: Self, c: &Converter) -> Self {
            c.apply(f64::from(x)) as f32
        }
    }

    impl Part for f64 {
        const COMPLEX: DType = DType::C128;
        fn scaled(x: Self, c: &Converter) -> Self {
            c.apply(x)
        }
    }
}

/// `|x|` for signed and unsigned integers (`None` for the minimum of a signed type).
trait IntAbs: Copy {
    fn int_abs(self) -> Option<Self>;
}

macro_rules! int_abs_signed {
    ($($t:ty),*) => {$(impl IntAbs for $t { fn int_abs(self) -> Option<Self> { self.checked_abs() } })*};
}
macro_rules! int_abs_unsigned {
    ($($t:ty),*) => {$(impl IntAbs for $t { fn int_abs(self) -> Option<Self> { Some(self) } })*};
}
int_abs_signed!(i8, i16, i32, i64);
int_abs_unsigned!(u8, u16, u32, u64);

fn int_abs<T: IntAbs>(x: T) -> Option<T> {
    x.int_abs()
}

/// `x^e` elementwise, with the unit's power (`e` whole, or `1/n` for a root; any other
/// exponent needs a dimensionless unit).
fn pow_plan(e: f64, unit: Option<&Unit>) -> Result<(Option<Converter>, Option<Unit>)> {
    let Some(u) = unit else {
        return Ok((None, None));
    };
    #[allow(clippy::cast_possible_truncation)]
    if e.fract() == 0.0 && e.abs() <= 64.0 {
        return Ok((None, Some(u.powi(e as i32)?)));
    }
    let r = 1.0 / e;
    #[allow(clippy::cast_possible_truncation)]
    if (r - r.round()).abs() < 1e-12 && r.abs() <= 64.0 && r.round() != 0.0 {
        return Ok((None, Some(u.root(r.round() as i32)?)));
    }
    let c = u
        .converter_to(&Unit::one())
        .map_err(|_| UnitError::UnsupportedOp {
            op: "a fractional power",
            unit: u.to_string(),
        })?;
    Ok((Some(c), None))
}

fn pow_values(
    t: &TypedArray<crate::Owned>,
    e: f64,
    conv: Option<&Converter>,
) -> Result<TypedArray<crate::Owned>> {
    match t {
        TypedArray::F32(a) if conv.is_none() => {
            #[allow(clippy::cast_possible_truncation)]
            let e32 = e as f32;
            Ok(TypedArray::F32(a.mapv(|x| x.powf(e32))))
        }
        // The principal value, as numpy's `power` of a complex base.
        TypedArray::C64(a) => {
            #[allow(clippy::cast_possible_truncation)]
            let e32 = e as f32;
            Ok(TypedArray::C64(
                complex_scaled(a, conv)?.mapv(|z| z.powf(e32)),
            ))
        }
        TypedArray::C128(a) => Ok(TypedArray::C128(
            complex_scaled(a, conv)?.mapv(|z| z.powf(e)),
        )),
        TypedArray::Bool(_)
        | TypedArray::Str(_)
        | TypedArray::DateTime(_)
        | TypedArray::CfDate(_)
        | TypedArray::TimeDelta(_) => Err(CoreError::UnsupportedDType {
            op: "power",
            dtype: t.dtype(),
        }),
        other => {
            let x = crate::ops::to_f64(&other.view(), conv)?;
            Ok(TypedArray::F64(x.mapv(|v| v.powf(e))))
        }
    }
}

impl<S: Repr> VariableBase<S> {
    /// [`DataArrayBase::math`] of the values.
    ///
    /// # Errors
    /// As for [`DataArrayBase::math`].
    pub fn math(&self, f: MathFn) -> Result<Variable> {
        let (conv, unit) = plan(f, self.unit())?;
        self.mapped(f.name(), unit, move |t| compute(f, &t, conv.as_ref()))
    }

    /// [`DataArrayBase::pow`] of the values.
    ///
    /// # Errors
    /// As for [`DataArrayBase::pow`].
    pub fn pow(&self, e: f64) -> Result<Variable> {
        let (conv, unit) = pow_plan(e, self.unit())?;
        self.mapped("power", unit, move |t| pow_values(&t, e, conv.as_ref()))
    }

    /// `f` of every block (lazily for lazy data), with `unit` and the attributes.
    fn mapped(
        &self,
        what: &'static str,
        unit: Option<Unit>,
        f: impl Fn(TypedArray<crate::Owned>) -> Result<TypedArray<crate::Owned>> + Send + Sync + 'static,
    ) -> Result<Variable> {
        let mut out = if self.is_lazy() {
            self.lazy_map(what, f)?
        } else {
            let mut v =
                Variable::from_typed(self.dims().iter().cloned(), f(self.typed()?.to_owned())?)?;
            v.attrs_mut()
                .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
            v
        };
        out.set_unit(unit);
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// `f` of every value, with its unit rule; see the module notes.
    ///
    /// # Errors
    /// [`UnitError::UnsupportedOp`] for a unit the function does not accept;
    /// [`UnitError::InexactRoot`]; [`CoreError::UnsupportedDType`] for non-numeric data;
    /// [`CoreError::ArithmeticOverflow`] for integers.
    pub fn math(&self, f: MathFn) -> Result<DataArray> {
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().math(f)?,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }

    /// Every value to the power `e` (`f64` result; `f32` stays `f32`). The unit is raised
    /// to `e` when it is whole, rooted when it is `1/n`; otherwise the data must be
    /// dimensionless.
    ///
    /// # Errors
    /// As for [`math`](Self::math).
    pub fn pow(&self, e: f64) -> Result<DataArray> {
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().pow(e)?,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }
}

/// `-x`: negation with its unit rule ([`MathFn::Neg`]).
impl<S: Repr> std::ops::Neg for &DataArrayBase<S> {
    type Output = Result<DataArray>;
    fn neg(self) -> Result<DataArray> {
        self.math(MathFn::Neg)
    }
}
