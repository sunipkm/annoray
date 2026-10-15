//! Missing values: [`isnull`](VariableBase::isnull), [`fillna`](VariableBase::fillna),
//! [`where_`](DataArrayBase::where_), [`dropna`](DataArrayBase::dropna) and
//! [`reindex`](DataArrayBase::reindex).
//!
//! "Missing" is NaN for floats and NaT for times; other types have no missing value.

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};

use crate::align::{Join, Obj, apply_array, apply_dataset, plan};
use crate::dataarray::DataArrayBase;
use crate::ops::{broadcast_to_dims, to_f64};
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, NaT, Name, Owned, Repr, Result, TypedArray,
    Variable, VariableBase, View,
};

/// Which positions `dropna` drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum How {
    /// Positions with any missing value (the default).
    #[default]
    Any,
    /// Positions where every value is missing.
    All,
}

/// Where values are missing.
fn null_mask(t: &TypedArray<View<'_>>) -> ArrayD<bool> {
    match t {
        TypedArray::F32(a) => a.mapv(f32::is_nan),
        TypedArray::F64(a) => a.mapv(f64::is_nan),
        TypedArray::DateTime(a) => a.map(NaT::is_nat),
        TypedArray::TimeDelta(a) => a.map(NaT::is_nat),
        TypedArray::CfDate(a) => a.map(NaT::is_nat),
        other => ArrayD::from_elem(IxDyn(other.shape()), false),
    }
}

/// `v`'s metadata on new data.
fn like<S: Repr>(v: &VariableBase<S>, data: TypedArray<Owned>) -> Result<Variable> {
    let mut out = Variable::from_typed(v.dims().iter().cloned(), data)?;
    out.set_unit(v.unit().cloned());
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

impl<S: Repr> VariableBase<S> {
    /// Where values are missing (NaN or NaT), as booleans.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn isnull(&self) -> Result<Variable> {
        if self.is_lazy() {
            return self
                .lazy_map("isnull", |t| Ok(TypedArray::Bool(null_mask(&t.view()))))
                .map(|mut v| {
                    v.set_unit(None);
                    v.attrs_mut().clear();
                    v
                });
        }
        Variable::from_typed(
            self.dims().iter().cloned(),
            TypedArray::Bool(null_mask(&self.typed()?)),
        )
    }

    /// Where values are present.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn notnull(&self) -> Result<Variable> {
        if self.is_lazy() {
            return self
                .lazy_map("notnull", |t| {
                    Ok(TypedArray::Bool(null_mask(&t.view()).mapv(|m| !m)))
                })
                .map(|mut v| {
                    v.set_unit(None);
                    v.attrs_mut().clear();
                    v
                });
        }
        Variable::from_typed(
            self.dims().iter().cloned(),
            TypedArray::Bool(null_mask(&self.typed()?).mapv(|m| !m)),
        )
    }

    /// NaN replaced by `value` (other types have no NaN and are returned as they are).
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn fillna(&self, value: f64) -> Result<Variable> {
        if self.is_lazy() {
            return self.lazy_map("fillna", move |t| Ok(filled(&t.view(), value)));
        }
        let data = filled(&self.typed()?, value);
        like(self, data)
    }

    /// 1.0 where the value is present, 0.0 where it is missing, as `f64` (lazy data stays
    /// lazy); dimensionless.
    pub(crate) fn valid_f64(&self) -> Result<Variable> {
        let mask = |t: &TypedArray<View<'_>>| null_mask(t).mapv(|m| if m { 0.0 } else { 1.0 });
        let mut v = if self.is_lazy() {
            self.lazy_map("valid", move |t| Ok(TypedArray::F64(mask(&t.view()))))?
        } else {
            Variable::from_typed(
                self.dims().iter().cloned(),
                TypedArray::F64(mask(&self.typed()?)),
            )?
        };
        v.set_unit(None);
        v.attrs_mut().clear();
        Ok(v)
    }

    /// `f` applied lazily, box by box (unit and attributes kept).
    fn lazy_map(
        &self,
        what: &'static str,
        f: impl Fn(TypedArray<Owned>) -> Result<TypedArray<Owned>> + Send + Sync + 'static,
    ) -> Result<Variable> {
        let src = crate::lazy::MapSource::new(self.as_lazy_source()?, what, f)?;
        let mut out = Variable::from_lazy(
            self.dims().iter().cloned(),
            crate::LazyArray::new(std::sync::Arc::new(src)),
        )?;
        out.set_unit(self.unit().cloned());
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }
}

/// NaNs replaced by `value` (other types unchanged).
fn filled(t: &TypedArray<View<'_>>, value: f64) -> TypedArray<Owned> {
    match t {
        #[allow(clippy::cast_possible_truncation)] // the fill value in the data's type
        TypedArray::F32(a) => {
            let v = value as f32;
            TypedArray::F32(a.mapv(|x| if x.is_nan() { v } else { x }))
        }
        TypedArray::F64(a) => TypedArray::F64(a.mapv(|x| if x.is_nan() { value } else { x })),
        other => other.to_owned(),
    }
}

/// Keep `data` where `cond` holds; elsewhere NaN (NaT for times) or `other`.
fn where_data(
    t: TypedArray<View<'_>>,
    cond: &ArrayViewD<'_, bool>,
    other: Option<f64>,
) -> Result<TypedArray<Owned>> {
    let pick = |keep: bool, x: f64| if keep { x } else { other.unwrap_or(f64::NAN) };
    Ok(match t {
        #[allow(clippy::cast_possible_truncation)] // `other` in the data's type
        TypedArray::F32(a) => {
            let o = other.map_or(f32::NAN, |o| o as f32);
            TypedArray::F32(
                ndarray::Zip::from(&a)
                    .and(cond)
                    .map_collect(|x, c| if *c { *x } else { o }),
            )
        }
        TypedArray::DateTime(a) if other.is_none() => TypedArray::DateTime(
            ndarray::Zip::from(&a)
                .and(cond)
                .map_collect(|x, c| if *c { *x } else { NaT::NAT }),
        ),
        TypedArray::TimeDelta(a) if other.is_none() => TypedArray::TimeDelta(
            ndarray::Zip::from(&a)
                .and(cond)
                .map_collect(|x, c| if *c { *x } else { NaT::NAT }),
        ),
        TypedArray::CfDate(a) if other.is_none() => TypedArray::CfDate(
            ndarray::Zip::from(&a)
                .and(cond)
                .map_collect(|x, c| if *c { *x } else { NaT::NAT }),
        ),
        t if !matches!(
            t.dtype(),
            DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
        ) =>
        {
            let f = to_f64(&t, None)?;
            TypedArray::F64(
                ndarray::Zip::from(&f)
                    .and(cond)
                    .map_collect(|x, c| pick(*c, *x)),
            )
        }
        t => {
            return Err(CoreError::UnsupportedDType {
                op: "where (needs a missing value or `other`)",
                dtype: t.dtype(),
            });
        }
    })
}

/// `where` computed for whichever box is read: the data's box, and the condition's part of it
/// (broadcast over dimensions it lacks).
#[derive(Debug)]
struct WhereSource {
    data: std::sync::Arc<dyn crate::LazySource>,
    dims: Vec<Name>,
    cond: std::sync::Arc<dyn crate::LazySource>,
    cdims: Vec<Name>,
    other: Option<f64>,
}

impl WhereSource {
    fn apply(&self, data: TypedArray<Owned>, cond: &ArrayD<bool>) -> Result<TypedArray<Owned>> {
        let c = broadcast_to_dims(cond.view(), &self.cdims, &self.dims)?;
        let c = c
            .broadcast(IxDyn(data.shape()))
            .ok_or_else(|| CoreError::InvalidReduction("cannot broadcast the condition".into()))?;
        where_data(data.view(), &c, self.other)
    }
}

/// [`WhereSource`] with its element type.
#[derive(Debug)]
struct TypedWhere(WhereSource, DType);

impl crate::LazySource for TypedWhere {
    fn dtype(&self) -> DType {
        self.1
    }
    fn shape(&self) -> &[usize] {
        self.0.data.shape()
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        let w = &self.0;
        let data = w.data.read_slice(start, count)?;
        let (mut cs, mut cc) = (Vec::new(), Vec::new());
        for d in &w.cdims {
            let i = w
                .dims
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| CoreError::NoSuchDim(d.clone()))?;
            cs.push(start.get(i).copied().unwrap_or(0));
            cc.push(count.get(i).copied().unwrap_or(0));
        }
        let cond = match w.cond.read_slice(&cs, &cc)? {
            TypedArray::Bool(c) => c,
            other => {
                return Err(CoreError::DTypeMismatch {
                    expected: DType::Bool,
                    found: other.dtype(),
                });
            }
        };
        w.apply(data, &cond)
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        self.0.data.chunking(axis)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// Where values are missing, as booleans (coordinates kept).
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn isnull(&self) -> Result<DataArray> {
        self.with_data(self.variable().isnull()?)
    }

    /// Where values are present.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn notnull(&self) -> Result<DataArray> {
        self.with_data(self.variable().notnull()?)
    }

    /// NaN replaced by `value`.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn fillna(&self, value: f64) -> Result<DataArray> {
        self.with_data(self.variable().fillna(value)?)
    }

    fn with_data(&self, v: Variable) -> Result<DataArray> {
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(self.name().cloned(), v, coords))
    }

    /// Values where `cond` is true, NaN (NaT for times) elsewhere; integers become `f64`.
    /// `cond` is aligned with this array (inner join) and broadcast over the dimensions it
    /// lacks.
    ///
    /// # Errors
    /// [`CoreError::DTypeMismatch`] if `cond` is not boolean; [`CoreError::InvalidReduction`]
    /// if `cond` has dimensions this array lacks; [`CoreError::UnsupportedDType`] for strings
    /// and booleans; alignment errors.
    pub fn where_<R: Repr>(&self, cond: &DataArrayBase<R>) -> Result<DataArray> {
        self.where_impl(cond, None)
    }

    /// Values where `cond` is true, `other` elsewhere (as `f64`, or `f32` for `f32` data).
    ///
    /// # Errors
    /// As for [`where_`](Self::where_).
    pub fn where_or<R: Repr>(&self, cond: &DataArrayBase<R>, other: f64) -> Result<DataArray> {
        self.where_impl(cond, Some(other))
    }

    fn where_impl<R: Repr>(
        &self,
        cond: &DataArrayBase<R>,
        other: Option<f64>,
    ) -> Result<DataArray> {
        let p = plan(&[Obj::of_array(self), Obj::of_array(cond)], Join::Inner)?;
        let (data, cond) = (apply_array(self, &p, 0)?, apply_array(cond, &p, 1)?);
        if let Some(d) = cond.dims().iter().find(|d| !data.dims().contains(d)) {
            return Err(CoreError::InvalidReduction(format!(
                "the condition has dimension {d:?}, which the data lacks"
            )));
        }
        if cond.dtype() != DType::Bool {
            return Err(CoreError::DTypeMismatch {
                expected: DType::Bool,
                found: cond.dtype(),
            });
        }
        if data.variable().is_lazy() || cond.variable().is_lazy() {
            let src = WhereSource {
                data: data.variable().as_lazy_source()?,
                dims: data.dims().to_vec(),
                cond: cond.variable().as_lazy_source()?,
                cdims: cond.dims().to_vec(),
                other,
            };
            // The element type, from the same selection on empty data.
            let probe = src.apply(
                TypedArray::empty(src.data.dtype(), &vec![0; src.dims.len().max(1)])
                    .ok_or_else(|| CoreError::Read("where: empty probe".into()))?,
                &ArrayD::from_elem(IxDyn(&vec![0; src.cdims.len()]), true),
            )?;
            let dtype = probe.dtype();
            let mut v = Variable::from_lazy(
                data.dims().iter().cloned(),
                crate::LazyArray::new(std::sync::Arc::new(TypedWhere(src, dtype))),
            )?;
            v.set_unit(data.unit().cloned());
            v.attrs_mut()
                .extend(data.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
            return data.with_data(v);
        }
        let c = cond.data::<bool>()?;
        let c = broadcast_to_dims(c, cond.dims(), data.dims())?;
        let c = c
            .broadcast(IxDyn(data.shape()))
            .ok_or_else(|| CoreError::InvalidReduction("cannot broadcast the condition".into()))?;
        let out = like(
            data.variable(),
            where_data(data.variable().typed()?, &c, other)?,
        )?;
        data.with_data(out)
    }

    /// Positions along `dim` with missing values dropped (`How::Any`: any missing value at
    /// that position; `How::All`: all of them).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; read errors for lazy data.
    pub fn dropna(&self, dim: &str, how: How) -> Result<DataArray> {
        let axis = self.variable().axis(dim)?;
        let keep = keep_positions(&[(null_mask(&self.variable().typed()?), axis)], how);
        self.take(dim, &keep?)
    }

    /// Reindexed onto `labels` along `dim` (a 1-D variable of coordinate values): positions
    /// for labels it lacks are NaN (NaT for times; integers become `f64`).
    ///
    /// # Errors
    /// Alignment errors (coordinate types or units differ, duplicate labels).
    pub fn reindex<R: Repr>(&self, dim: &str, labels: &VariableBase<R>) -> Result<DataArray> {
        let dummy = target(dim, labels)?;
        let p = plan(&[Obj::of_array(self), Obj::of_array(&dummy)], Join::Right)?;
        apply_array(self, &p, 0)
    }

    /// Reindexed onto the dimension coordinates of `other` (for the dimensions both have).
    ///
    /// # Errors
    /// As for [`reindex`](Self::reindex).
    pub fn reindex_like<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<DataArray> {
        let p = plan(&[Obj::of_array(self), Obj::of_array(other)], Join::Right)?;
        apply_array(self, &p, 0)
    }
}

/// A data array holding only the coordinate `labels` along `dim`.
fn target<R: Repr>(dim: &str, labels: &VariableBase<R>) -> Result<DataArray> {
    let n = labels.len();
    if labels.ndim() != 1 {
        return Err(CoreError::InvalidReduction(
            "reindex labels must be one-dimensional".into(),
        ));
    }
    let dim = Name::from(dim);
    let mut coord = Variable::from_typed([dim.clone()], labels.typed()?.to_owned())?;
    coord.set_unit(labels.unit().cloned());
    let data = Variable::new([dim.clone()], ArrayD::from_elem(IxDyn(&[n]), 0.0))?;
    DataArray::new(data).with_coord(dim, coord)
}

/// Positions along each mask's axis to keep (all masks share that axis's length).
fn keep_positions(masks: &[(ArrayD<bool>, usize)], how: How) -> Result<Vec<isize>> {
    let n = masks.first().map_or(0, |(m, a)| m.len_of(Axis(*a)));
    let mut keep = Vec::new();
    for i in 0..n {
        let (mut any, mut all) = (false, true);
        for (m, a) in masks {
            let lane = m.index_axis(Axis(*a), i);
            any |= lane.iter().any(|x| *x);
            all &= lane.iter().all(|x| *x);
        }
        let drop = match how {
            How::Any => any,
            How::All => all,
        };
        if !drop {
            keep.push(isize::try_from(i).map_err(|_| CoreError::ArithmeticOverflow("dropna"))?);
        }
    }
    Ok(keep)
}

impl<S: Repr> DatasetBase<S> {
    /// NaN replaced by `value` in every data variable.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn fillna(&self, value: f64) -> Result<Dataset> {
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(k.clone(), c.to_owned())?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), v.fillna(value)?)?;
        }
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }

    /// Positions along `dim` dropped where data variables with that dimension have missing
    /// values (`How::Any`: any of them; `How::All`: all values of all of them).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; read errors for lazy data.
    pub fn dropna(&self, dim: &str, how: How) -> Result<Dataset> {
        if !self.sizes().contains_key(dim) {
            return Err(CoreError::NoSuchDim(dim.into()));
        }
        let mut masks = Vec::new();
        for v in self.data_vars().values() {
            if let Ok(axis) = v.axis(dim) {
                masks.push((null_mask(&v.typed()?), axis));
            }
        }
        if masks.is_empty() {
            return Ok(self.to_owned());
        }
        self.take(dim, &keep_positions(&masks, how)?)
    }

    /// Reindexed onto `labels` along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::reindex`].
    pub fn reindex<R: Repr>(&self, dim: &str, labels: &VariableBase<R>) -> Result<Dataset> {
        let dummy = target(dim, labels)?;
        let p = plan(&[Obj::of_dataset(self), Obj::of_array(&dummy)], Join::Right)?;
        apply_dataset(self, &p, 0)
    }
}
