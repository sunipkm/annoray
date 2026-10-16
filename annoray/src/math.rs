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
//! - Types as numpy: `f32` stays `f32`; integers stay integers for `Neg`, `Abs` and `Square`
//!   (checked; overflow is an error), and become `f64` otherwise; booleans and non-numeric
//!   data are refused. NaN propagates.
//! - Lazy data stays lazy (applied block by block when read).

use annoray_units::{Converter, Unit, UnitError, UnitKind};
use ndarray::ArrayD;

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
        MathFn::Floor | MathFn::Ceil | MathFn::Trunc | MathFn::Rint => (None, Some(u.clone())),
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
    let keeps_int = matches!(f, MathFn::Neg | MathFn::Abs | MathFn::Square);
    macro_rules! int {
        ($a:expr, $t:ty, $var:ident) => {{
            if keeps_int && conv.is_none() {
                let mut out = $a.clone();
                for x in &mut out {
                    let v: Option<$t> = match f {
                        MathFn::Neg => (0 as $t).checked_sub(*x),
                        MathFn::Abs => int_abs(*x),
                        _ => x.checked_mul(*x),
                    };
                    *x = v.ok_or(CoreError::ArithmeticOverflow("integer math"))?;
                }
                return Ok(TypedArray::$var(out));
            }
        }};
    }
    match t {
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
