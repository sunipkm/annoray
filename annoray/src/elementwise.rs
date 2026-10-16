//! Elementwise operations: `clip`, `round`, `isin`, and the tensor contraction `dot`.
//!
//! - `clip(min, max)` is numpy's `minimum(maximum(x, min), max)` (NaN data or bounds give
//!   NaN; `min > max` gives `max`). Bounds are numbers in the data's unit, quantities in any
//!   unit of the same kind (converted, absolute temperatures included), or data arrays
//!   (broadcast and aligned like arithmetic, converted likewise). Scalar bounds keep the
//!   data's type (`f32`, integers when the bound is whole); attributes and unit are kept.
//! - `round(decimals)` is numpy's: half to even, `decimals < 0` rounding to tens, hundreds…
//!   (integers too, exactly); units and attributes kept.
//! - `isin(labels)` is a boolean array: numbers compare in the data's unit (a quantity is
//!   converted first; one against data without a unit is an error), strings, dates and
//!   durations compare exactly; labels of another kind match nothing. Unit and attributes
//!   are dropped, as xarray.
//! - `dot` multiplies (units multiply) and sums over the dimensions shared by more than one
//!   operand (or the ones named), as `xr.dot`/`numpy.einsum`: NaN propagates.
//! - Lazy data stays lazy (`clip`, `round`, `isin`; `dot` streams its sum).

use ndarray::ArrayD;

use crate::{
    BinOp, CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Label, NaT, Name,
    ReduceOptions, Reduction, Repr, Result, TypedArray, Variable, VariableBase,
};
use annoray_units::{Unit, UnitPolicy};

/// A bound of [`clip`](DataArrayBase::clip).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ClipBound {
    /// A number in the data's unit.
    Value(f64),
    /// A number in another unit of the same kind (converted).
    Quantity(f64, Unit),
    /// An array, broadcast and aligned against the data (its unit converted).
    Array(Box<DataArray>),
}

impl From<f64> for ClipBound {
    fn from(v: f64) -> Self {
        Self::Value(v)
    }
}

impl From<DataArray> for ClipBound {
    fn from(a: DataArray) -> Self {
        Self::Array(Box::new(a))
    }
}

/// `v` as a 0-d array of type `dtype` when it is representable exactly; `None` otherwise.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn scalar_as(dtype: DType, v: f64) -> Option<TypedArray<crate::Owned>> {
    let whole = v.fract() == 0.0 && v.abs() < 9.0e15;
    let int = || whole.then_some(v as i64);
    fn a<T: Clone>(x: T) -> ArrayD<T> {
        ArrayD::from_elem(ndarray::IxDyn(&[]), x)
    }
    Some(match dtype {
        DType::F64 => TypedArray::F64(a(v)),
        DType::F32 if f64::from(v as f32) == v || v.is_nan() || v.is_infinite() => {
            TypedArray::F32(a(v as f32))
        }
        DType::I8 => TypedArray::I8(a(i8::try_from(int()?).ok()?)),
        DType::I16 => TypedArray::I16(a(i16::try_from(int()?).ok()?)),
        DType::I32 => TypedArray::I32(a(i32::try_from(int()?).ok()?)),
        DType::I64 => TypedArray::I64(a(int()?)),
        DType::U8 => TypedArray::U8(a(u8::try_from(int()?).ok()?)),
        DType::U16 => TypedArray::U16(a(u16::try_from(int()?).ok()?)),
        DType::U32 => TypedArray::U32(a(u32::try_from(int()?).ok()?)),
        DType::U64 => TypedArray::U64(a(u64::try_from(int()?).ok()?)),
        _ => TypedArray::F64(a(v)),
    })
}

fn bound_array<S: Repr>(da: &DataArrayBase<S>, b: &ClipBound) -> Result<DataArray> {
    let scalar = |v: f64, unit: Option<Unit>| -> Result<DataArray> {
        let t = scalar_as(da.variable().dtype(), v)
            .unwrap_or_else(|| TypedArray::F64(ArrayD::from_elem(ndarray::IxDyn(&[]), v)));
        let mut var = Variable::from_typed(Vec::<Name>::new(), t)?;
        var.set_unit(unit);
        Ok(DataArray::new(var))
    };
    match b {
        ClipBound::Value(v) => scalar(*v, da.unit().cloned()),
        ClipBound::Quantity(v, u) => {
            // Converted here so that the bound can keep the data's type.
            match da.unit() {
                Some(to) => scalar(u.converter_to(to)?.apply(*v), Some(to.clone())),
                None => scalar(*v, Some(u.clone())),
            }
        }
        ClipBound::Array(a) => Ok((**a).clone()),
    }
}

/// numpy's `round` of an integer to `10^-decimals` (`decimals < 0`), half to even.
fn round_int(x: i128, step: i128) -> Option<i128> {
    let q = x.div_euclid(step);
    let r = x.rem_euclid(step);
    let twice = r.checked_mul(2)?;
    let up = twice > step || (twice == step && q.rem_euclid(2) == 1);
    q.checked_add(i128::from(up))?.checked_mul(step)
}

fn rounded(t: &TypedArray<crate::Owned>, decimals: i32) -> Result<TypedArray<crate::Owned>> {
    let p = |d: i32| 10f64.powi(d);
    macro_rules! float {
        ($a:expr, $t:ty) => {{
            #[allow(clippy::cast_possible_truncation)]
            let (scale, up) = (p(decimals.abs()) as $t, decimals >= 0);
            $a.mapv(|x: $t| {
                if up {
                    (x * scale).round_ties_even() / scale
                } else {
                    (x / scale).round_ties_even() * scale
                }
            })
        }};
    }
    macro_rules! int {
        ($a:expr, $t:ty) => {{
            let step = 10i128
                .checked_pow(decimals.unsigned_abs())
                .ok_or(CoreError::ArithmeticOverflow("round"))?;
            let mut out = $a.clone();
            for x in &mut out {
                *x = round_int(i128::from(*x), step)
                    .and_then(|v| <$t>::try_from(v).ok())
                    .ok_or(CoreError::ArithmeticOverflow("round"))?;
            }
            out
        }};
    }
    Ok(match t {
        TypedArray::F64(a) => TypedArray::F64(float!(a, f64)),
        TypedArray::F32(a) => TypedArray::F32(float!(a, f32)),
        other if decimals >= 0 => other.clone(),
        TypedArray::I8(a) => TypedArray::I8(int!(a, i8)),
        TypedArray::I16(a) => TypedArray::I16(int!(a, i16)),
        TypedArray::I32(a) => TypedArray::I32(int!(a, i32)),
        TypedArray::I64(a) => TypedArray::I64(int!(a, i64)),
        TypedArray::U8(a) => TypedArray::U8(int!(a, u8)),
        TypedArray::U16(a) => TypedArray::U16(int!(a, u16)),
        TypedArray::U32(a) => TypedArray::U32(int!(a, u32)),
        TypedArray::U64(a) => TypedArray::U64(int!(a, u64)),
        other => {
            return Err(CoreError::UnsupportedDType {
                op: "round",
                dtype: other.dtype(),
            });
        }
    })
}

/// Test elements, by the kind they match.
#[derive(Default, Clone)]
struct Wanted {
    numbers: Vec<f64>,
    strings: Vec<String>,
    times: Vec<chrono::NaiveDateTime>,
    cf: Vec<annoray_units::CfDate>,
    deltas: Vec<chrono::TimeDelta>,
}

fn wanted(labels: &[Label], unit: Option<&Unit>) -> Result<Wanted> {
    let mut w = Wanted::default();
    for l in labels {
        #[allow(clippy::cast_precision_loss)]
        match l {
            Label::Int(v) => w.numbers.push(*v as f64),
            Label::Float(v) => w.numbers.push(*v),
            Label::Quantity(v, u) => {
                let to = unit.ok_or_else(|| {
                    CoreError::InvalidReduction(format!(
                        "isin: {v} {u} against data without a unit"
                    ))
                })?;
                w.numbers.push(u.converter_to(to)?.apply(*v));
            }
            Label::Str(s) => w.strings.push(s.clone()),
            Label::DateTime(d) => w.times.push(*d),
            Label::CfDate(d) => w.cf.push(*d),
            Label::TimeDelta(d) => w.deltas.push(*d),
        }
    }
    Ok(w)
}

fn membership(t: &TypedArray<crate::Owned>, w: &Wanted) -> ArrayD<bool> {
    #[allow(clippy::cast_precision_loss)]
    let num = |x: f64| w.numbers.contains(&x);
    match t {
        TypedArray::F64(a) => a.mapv(num),
        TypedArray::F32(a) => a.mapv(|x| num(f64::from(x))),
        TypedArray::I8(a) => a.mapv(|x| num(f64::from(x))),
        TypedArray::I16(a) => a.mapv(|x| num(f64::from(x))),
        TypedArray::I32(a) => a.mapv(|x| num(f64::from(x))),
        #[allow(clippy::cast_precision_loss)]
        TypedArray::I64(a) => a.mapv(|x| num(x as f64)),
        TypedArray::U8(a) => a.mapv(|x| num(f64::from(x))),
        TypedArray::U16(a) => a.mapv(|x| num(f64::from(x))),
        TypedArray::U32(a) => a.mapv(|x| num(f64::from(x))),
        #[allow(clippy::cast_precision_loss)]
        TypedArray::U64(a) => a.mapv(|x| num(x as f64)),
        TypedArray::Bool(a) => a.mapv(|x| num(if x { 1.0 } else { 0.0 })),
        TypedArray::Str(a) => a.map(|x| w.strings.contains(x)),
        TypedArray::DateTime(a) => a.mapv(|x| !x.is_nat() && w.times.contains(&x)),
        TypedArray::CfDate(a) => a.mapv(|x| !x.is_nat() && w.cf.contains(&x)),
        TypedArray::TimeDelta(a) => a.mapv(|x| !x.is_nat() && w.deltas.contains(&x)),
    }
}

impl<S: Repr> VariableBase<S> {
    /// [`DataArrayBase::round`] of the values.
    ///
    /// # Errors
    /// As for [`DataArrayBase::round`].
    pub fn round(&self, decimals: i32) -> Result<Variable> {
        if self.is_lazy() {
            return self.lazy_map("round", move |t| rounded(&t, decimals));
        }
        let mut out = Variable::from_typed(
            self.dims().iter().cloned(),
            rounded(&self.typed()?.to_owned(), decimals)?,
        )?;
        out.set_unit(self.unit().cloned());
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }

    /// [`DataArrayBase::isin`] of the values.
    ///
    /// # Errors
    /// As for [`DataArrayBase::isin`].
    pub fn isin(&self, labels: &[Label]) -> Result<Variable> {
        let w = wanted(labels, self.unit())?;
        let mut out = if self.is_lazy() {
            self.lazy_map("isin", move |t| Ok(TypedArray::Bool(membership(&t, &w))))?
        } else {
            Variable::from_typed(
                self.dims().iter().cloned(),
                TypedArray::Bool(membership(&self.typed()?.to_owned(), &w)),
            )?
        };
        out.set_unit(None);
        out.attrs_mut().clear();
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// Limit the values to `[min, max]` (either may be `None`); see the module notes. With
    /// neither bound the data is returned unchanged.
    ///
    /// # Errors
    /// Unit errors (a bound of another kind, a bare-number array bound against data with a
    /// unit); alignment errors for array bounds; [`CoreError::UnsupportedDType`] for
    /// non-numeric data.
    pub fn clip(&self, min: Option<ClipBound>, max: Option<ClipBound>) -> Result<DataArray> {
        let mut out = self.to_owned();
        for (b, op) in [(min, BinOp::Max), (max, BinOp::Min)] {
            if let Some(b) = b {
                let rhs = bound_array(self, &b)?;
                out = out.binary(op, &rhs, UnitPolicy::AutoRescale)?;
            }
        }
        if let Some(n) = self.name() {
            out = out.with_name(n.clone());
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// Round to `decimals` (negative: to tens, hundreds…), numpy's half-to-even rounding.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for non-numeric data; [`CoreError::ArithmeticOverflow`]
    /// when rounding an integer leaves its type.
    pub fn round(&self, decimals: i32) -> Result<DataArray> {
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().round(decimals)?,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }

    /// Whether each value is one of `labels` (see the module notes).
    ///
    /// # Errors
    /// A quantity label against data without a unit, or of another kind.
    pub fn isin(&self, labels: &[Label]) -> Result<DataArray> {
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().isin(labels)?,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }

    /// The product with `other` summed over their shared dimensions (or over `dims`); see
    /// [`dot`].
    ///
    /// # Errors
    /// As for [`dot`].
    pub fn dot<R: Repr>(
        &self,
        other: &DataArrayBase<R>,
        dims: Option<&[&str]>,
    ) -> Result<DataArray> {
        dot(&[&self.to_owned(), &other.to_owned()], dims)
    }
}

/// The product of `arrays` summed over `dims`, or by default over every dimension that more
/// than one of them has (`xr.dot`). Units multiply; NaN propagates (no skipping); integers
/// stay integers.
///
/// # Errors
/// [`CoreError::NoSuchDim`] for a dimension none has; unit errors of multiplication;
/// alignment errors; [`CoreError::InvalidReduction`] without arrays.
pub fn dot(arrays: &[&DataArray], dims: Option<&[&str]>) -> Result<DataArray> {
    let (first, rest) = arrays
        .split_first()
        .ok_or_else(|| CoreError::InvalidReduction("dot needs at least one array".into()))?;
    let sum_over: Vec<Name> = match dims {
        Some(d) => {
            let all: Vec<&Name> = arrays.iter().flat_map(|a| a.dims()).collect();
            for x in d {
                if !all.iter().any(|n| *n == x) {
                    return Err(CoreError::NoSuchDim(Name::from(*x)));
                }
            }
            d.iter().map(|x| Name::from(*x)).collect()
        }
        None => {
            let mut seen: Vec<(Name, usize)> = Vec::new();
            for a in arrays {
                for d in a.dims() {
                    match seen.iter_mut().find(|(n, _)| n == d) {
                        Some((_, c)) => *c = c.saturating_add(1),
                        None => seen.push((d.clone(), 1)),
                    }
                }
            }
            seen.into_iter()
                .filter(|(_, c)| *c > 1)
                .map(|(n, _)| n)
                .collect()
        }
    };
    let mut prod = (*first).clone();
    for a in rest {
        prod = prod.try_mul(*a)?;
    }
    if sum_over.is_empty() {
        return Ok(prod);
    }
    prod.reduce(
        Reduction::Sum,
        sum_over,
        &ReduceOptions::default().skipna(false),
    )
}

fn per_data_var<S: Repr>(
    ds: &DatasetBase<S>,
    f: impl Fn(&DataArrayBase<crate::View<'_>>) -> Result<Option<Variable>>,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    for (k, c) in ds.coords() {
        out.add_coord(k.clone(), c.to_owned())?;
    }
    for k in ds.data_vars().keys() {
        let da = ds.get(k)?;
        let v = match f(&da)? {
            Some(v) => v,
            None => da.variable().to_owned(),
        };
        out.add_var(k.clone(), v)?;
    }
    *out.attrs_mut() = ds.attrs().clone();
    Ok(out)
}

fn numeric(d: DType) -> bool {
    !matches!(
        d,
        DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
    )
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::clip`] of every numeric data variable (others unchanged). Array
    /// bounds apply to every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::clip`].
    pub fn clip(&self, min: Option<ClipBound>, max: Option<ClipBound>) -> Result<Dataset> {
        per_data_var(self, |da| {
            if !numeric(da.variable().dtype()) {
                return Ok(None);
            }
            Ok(Some(
                da.clip(min.clone(), max.clone())?.variable().to_owned(),
            ))
        })
    }

    /// [`DataArrayBase::round`] of every numeric data variable (others unchanged).
    ///
    /// # Errors
    /// As for [`DataArrayBase::round`].
    pub fn round(&self, decimals: i32) -> Result<Dataset> {
        per_data_var(self, |da| {
            if !numeric(da.variable().dtype()) {
                return Ok(None);
            }
            Ok(Some(da.variable().round(decimals)?))
        })
    }

    /// [`DataArrayBase::isin`] of every data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::isin`].
    pub fn isin(&self, labels: &[Label]) -> Result<Dataset> {
        per_data_var(self, |da| Ok(Some(da.variable().isin(labels)?)))
    }
}
