//! Element-type conversion (xarray's `astype`) and constant arrays shaped like others
//! (`full_like`).
//!
//! Numbers and booleans convert to one another. Unlike numpy's unchecked casts, a value with
//! no counterpart in the target type is an error rather than an arbitrary result:
//!
//! - float → integer truncates toward zero, as numpy does; NaN, ±∞ and values outside the
//!   target's range are errors (numpy gives `INT_MIN`, saturates or wraps).
//! - integer → integer must fit (numpy wraps: 300 as `uint8` is 44).
//! - `f64` → `f32` rounds; a finite value beyond `f32`'s range is an error (numpy gives ∞).
//! - integer → float rounds to the nearest representable value, as numpy does.
//! - → `bool` is "nonzero" (NaN is true, as in numpy); `bool` → number is 0 or 1.
//!
//! The unit is kept, except for a boolean result, which has none. Converting to the current
//! type returns a copy. Lazy data stays lazy: it is converted block by block when read.
//! Strings, dates and durations do not convert.

use ndarray::ArrayD;

use crate::{
    CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Element, Owned, Repr, Result,
    TypedArray, Variable, VariableBase, View,
};

/// A source value, widened without loss.
#[derive(Clone, Copy)]
enum Wide {
    Int(i128),
    Float(f64),
}

impl std::fmt::Display for Wide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
        }
    }
}

/// A target element type of [`astype`](VariableBase::astype).
trait CastTo: Element + Copy {
    /// The value as `Self`, or why it has none.
    fn cast(v: Wide) -> std::result::Result<Self, &'static str>;
}

/// `x` truncated toward zero, as an integer, if it is finite and within `i128`.
#[allow(clippy::cast_possible_truncation)] // range checked first
fn trunc_int(x: f64) -> std::result::Result<i128, &'static str> {
    if x.is_nan() {
        return Err("NaN has no integer value");
    }
    if x.is_infinite() {
        return Err("an infinite value has no integer value");
    }
    let t = x.trunc();
    // 2^127, exactly representable.
    let bound = 170_141_183_460_469_231_731_687_303_715_884_105_728.0_f64;
    if t >= bound || t < -bound {
        return Err("out of range");
    }
    Ok(t as i128)
}

macro_rules! cast_int {
    ($($t:ty),*) => {$(
        impl CastTo for $t {
            fn cast(v: Wide) -> std::result::Result<Self, &'static str> {
                let i = match v {
                    Wide::Int(i) => i,
                    Wide::Float(x) => trunc_int(x)?,
                };
                <$t>::try_from(i).map_err(|_| "out of range")
            }
        }
    )*};
}
cast_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl CastTo for f64 {
    #[allow(clippy::cast_precision_loss)] // rounds to nearest, as numpy
    fn cast(v: Wide) -> std::result::Result<Self, &'static str> {
        Ok(match v {
            Wide::Int(i) => i as f64,
            Wide::Float(x) => x,
        })
    }
}

impl CastTo for f32 {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // range checked
    fn cast(v: Wide) -> std::result::Result<Self, &'static str> {
        let x = match v {
            Wide::Int(i) => return Ok(i as f32),
            Wide::Float(x) => x,
        };
        if x.is_finite() && x.abs() > f64::from(f32::MAX) {
            return Err("out of range");
        }
        Ok(x as f32)
    }
}

impl CastTo for bool {
    fn cast(v: Wide) -> std::result::Result<Self, &'static str> {
        Ok(match v {
            Wide::Int(i) => i != 0,
            Wide::Float(x) => x != 0.0,
        })
    }
}

/// The values of `t`, widened, in logical order; `None` for strings and times.
fn widened<'a>(t: &'a TypedArray<View<'_>>) -> Option<Box<dyn Iterator<Item = Wide> + 'a>> {
    macro_rules! int {
        ($a:expr) => {
            Some(Box::new($a.iter().map(|x| Wide::Int(i128::from(*x)))))
        };
    }
    match t {
        TypedArray::F32(a) => Some(Box::new(a.iter().map(|x| Wide::Float(f64::from(*x))))),
        TypedArray::F64(a) => Some(Box::new(a.iter().map(|x| Wide::Float(*x)))),
        TypedArray::I8(a) => int!(a),
        TypedArray::I16(a) => int!(a),
        TypedArray::I32(a) => int!(a),
        TypedArray::I64(a) => int!(a),
        TypedArray::U8(a) => int!(a),
        TypedArray::U16(a) => int!(a),
        TypedArray::U32(a) => int!(a),
        TypedArray::U64(a) => int!(a),
        TypedArray::Bool(a) => int!(a),
        _ => None,
    }
}

fn cast_as<T: CastTo>(t: &TypedArray<View<'_>>) -> Result<TypedArray<Owned>> {
    let unsupported = || CoreError::UnsupportedDType {
        op: "astype",
        dtype: t.dtype(),
    };
    let values = widened(t).ok_or_else(unsupported)?;
    let mut out = Vec::with_capacity(t.len());
    for v in values {
        out.push(T::cast(v).map_err(|reason| CoreError::Cast {
            value: v.to_string(),
            dtype: T::DTYPE,
            reason,
        })?);
    }
    let a = ArrayD::from_shape_vec(ndarray::IxDyn(t.shape()), out)
        .map_err(|e| CoreError::InvalidReduction(format!("astype: {e}")))?;
    Ok(T::wrap(a))
}

/// `t` converted to `dtype` (see the module notes).
pub(crate) fn cast(t: &TypedArray<View<'_>>, dtype: DType) -> Result<TypedArray<Owned>> {
    if t.dtype() == dtype {
        return Ok(t.to_owned());
    }
    match dtype {
        DType::F32 => cast_as::<f32>(t),
        DType::F64 => cast_as::<f64>(t),
        DType::I8 => cast_as::<i8>(t),
        DType::I16 => cast_as::<i16>(t),
        DType::I32 => cast_as::<i32>(t),
        DType::I64 => cast_as::<i64>(t),
        DType::U8 => cast_as::<u8>(t),
        DType::U16 => cast_as::<u16>(t),
        DType::U32 => cast_as::<u32>(t),
        DType::U64 => cast_as::<u64>(t),
        DType::Bool => cast_as::<bool>(t),
        other => Err(CoreError::UnsupportedDType {
            op: "astype (to this type)",
            dtype: other,
        }),
    }
}

/// Whether `astype` can convert from `from` to `to`.
fn convertible(from: DType, to: DType) -> bool {
    let num = |d: DType| {
        !matches!(
            d,
            DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta
        )
    };
    from == to || (num(from) && num(to))
}

impl<S: Repr> VariableBase<S> {
    /// The values converted to `dtype`; unit and attributes kept (a boolean result has no
    /// unit), encoding dropped. Lazy data stays lazy, converted block by block when read.
    ///
    /// Numbers and booleans convert to one another; a value with no counterpart in `dtype`
    /// is an error, where numpy's unchecked cast gives an arbitrary result:
    ///
    /// - float → integer truncates toward zero; NaN, ±∞ and values outside the target's
    ///   range are errors (numpy gives `INT_MIN`, saturates or wraps).
    /// - integer → integer must fit (numpy wraps: 300 as `uint8` is 44).
    /// - `f64` → `f32` rounds; a finite value beyond `f32`'s range is an error.
    /// - integer → float rounds to the nearest representable value.
    /// - → `bool` is "nonzero" (NaN is true); `bool` → number is 0 or 1.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for strings, dates and durations (unless `dtype` is
    /// the current type); [`CoreError::Cast`] for a value with no counterpart in `dtype`.
    pub fn astype(&self, dtype: DType) -> Result<Variable> {
        if !convertible(self.dtype(), dtype) {
            // Name the side that is not a number.
            let numeric_source = convertible(self.dtype(), DType::F64);
            return Err(CoreError::UnsupportedDType {
                op: "astype",
                dtype: if numeric_source { dtype } else { self.dtype() },
            });
        }
        if dtype == self.dtype() {
            // Nothing to convert: lazy data stays lazy, unread.
            let mut out = self.to_owned().with_encoding(crate::Encoding::default());
            if dtype == DType::Bool {
                out.set_unit(None);
            }
            return Ok(out);
        }
        let mut out = if self.is_lazy() {
            self.lazy_map("astype", move |t: TypedArray<Owned>| cast(&t.view(), dtype))?
        } else {
            let mut v =
                Variable::from_typed(self.dims().iter().cloned(), cast(&self.typed()?, dtype)?)?;
            v.set_unit(self.unit().cloned());
            v.attrs_mut()
                .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
            v
        };
        if dtype == DType::Bool {
            out.set_unit(None);
        }
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// The values converted to `dtype`; coordinates, name and attributes kept. See
    /// [`VariableBase::astype`].
    ///
    /// # Errors
    /// As for [`VariableBase::astype`].
    pub fn astype(&self, dtype: DType) -> Result<DataArray> {
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().astype(dtype)?,
            coords,
        ))
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Every data variable converted to `dtype` (coordinates unchanged, as in xarray).
    ///
    /// # Errors
    /// As for [`VariableBase::astype`], for any data variable.
    pub fn astype(&self, dtype: DType) -> Result<Dataset> {
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(k.clone(), c.to_owned())?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), v.astype(dtype)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }
}

/// An array of `shape` holding `value` as `dtype`. Integer and boolean targets need the value
/// exactly (`0.5` is not an integer; a boolean is 0 or 1); floats round.
fn filled(dtype: DType, shape: &[usize], value: f64) -> Result<TypedArray<Owned>> {
    let exact = !matches!(dtype, DType::F32 | DType::F64);
    let reject = |reason| CoreError::Cast {
        value: value.to_string(),
        dtype,
        reason,
    };
    if exact && value.is_finite() && value.fract() != 0.0 {
        return Err(reject("not a whole number"));
    }
    if dtype == DType::Bool && value != 0.0 && value != 1.0 {
        return Err(reject("a boolean fill is 0 or 1"));
    }
    let one = ndarray::arr0(value).into_dyn();
    let scalar = cast(&TypedArray::<Owned>::F64(one).view(), dtype)?;
    let sh = ndarray::IxDyn(shape);
    macro_rules! fill {
        ($($v:ident),*) => {
            match scalar {
                $(TypedArray::$v(a) => match a.first() {
                    Some(x) => TypedArray::$v(ArrayD::from_elem(sh, *x)),
                    None => return Err(reject("no value")),
                },)*
                other => {
                    return Err(CoreError::UnsupportedDType {
                        op: "full_like",
                        dtype: other.dtype(),
                    });
                }
            }
        };
    }
    Ok(fill!(F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool))
}

impl<S: Repr> VariableBase<S> {
    /// A variable shaped like this one, every element `value` (in this variable's unit), of
    /// type `dtype` (this variable's by default). Attributes kept; a boolean result has no
    /// unit. Lazy data is not read.
    ///
    /// # Errors
    /// [`CoreError::Cast`] if `value` has no exact counterpart in an integer or boolean
    /// `dtype` (numpy truncates); [`CoreError::UnsupportedDType`] for strings and times.
    pub fn full_like(&self, value: f64, dtype: Option<DType>) -> Result<Variable> {
        let dtype = dtype.unwrap_or_else(|| self.dtype());
        let mut v = Variable::from_typed(
            self.dims().iter().cloned(),
            filled(dtype, self.shape(), value)?,
        )?;
        if dtype != DType::Bool {
            v.set_unit(self.unit().cloned());
        }
        v.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(v)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// A data array like this one (coordinates, name, attributes) filled with `value`; see
    /// [`VariableBase::full_like`].
    ///
    /// # Errors
    /// As for [`VariableBase::full_like`].
    pub fn full_like(&self, value: f64, dtype: Option<DType>) -> Result<DataArray> {
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().full_like(value, dtype)?,
            coords,
        ))
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Every data variable filled with `value` (coordinates unchanged); see
    /// [`VariableBase::full_like`].
    ///
    /// # Errors
    /// As for [`VariableBase::full_like`], for any data variable.
    pub fn full_like(&self, value: f64, dtype: Option<DType>) -> Result<Dataset> {
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(k.clone(), c.to_owned())?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), v.full_like(value, dtype)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }
}
