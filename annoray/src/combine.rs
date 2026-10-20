//! Combining objects: [`concat`] along a dimension and [`merge`] of datasets.
//!
//! The defaults are xarray's *new* combine defaults (`use_new_combine_kwarg_defaults`): other
//! dimensions must already be aligned ([`Join::Exact`]), only variables that have the
//! concatenation dimension are concatenated, and for everything else the first object wins
//! ([`Compat::Override`]). The older, more permissive behaviour (outer join, `equals` /
//! `no_conflicts` checks, concatenating every data variable) is available through the options.
//!
//! - **Units**: values in a compatible but different unit are converted to the first
//!   object's unit; a unit in one object and none in another is an error.
//! - **Element types**: mixed numeric types become `f64`; other mixes are an error.
//! - **Attributes** come from the first object (xarray's `combine_attrs = "override"`).

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::align::{Join, Obj, apply_dataset, plan};
use crate::array::{Marker, marker};
use crate::dataarray::DataArrayBase;
use crate::ops::{to_f64, variables_equal};
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Element, Name, Owned, Repr, Result,
    TypedArray, Variable, VariableBase, View,
};

/// Which variables to concatenate besides those that have the concatenation dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Concat {
    /// [`All`](Self::All) when the dimension is new, otherwise [`Minimal`](Self::Minimal)
    /// (the default for data variables).
    Auto,
    /// Only variables with the dimension (the default for coordinates).
    #[default]
    Minimal,
    /// Those whose values differ between objects, too.
    Different,
    /// All of them (each gains the dimension).
    All,
}

/// How to combine variables of the same name that are not concatenated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Compat {
    /// Take the first object's variable (the default).
    #[default]
    Override,
    /// Values (NaN equal to NaN), dimensions and units must be equal.
    Equals,
    /// As [`Equals`](Self::Equals), and attributes too.
    Identical,
    /// Where both have a non-NaN value it must be equal; NaNs are filled from the others.
    NoConflicts,
}

/// Options for [`concat()`] and [`concat_datasets`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConcatOptions {
    /// How to align the other dimensions.
    pub join: Join,
    /// Which data variables to concatenate.
    pub data_vars: Concat,
    /// Which coordinates to concatenate.
    pub coords: Concat,
    /// How to combine what is not concatenated.
    pub compat: Compat,
}

impl Default for ConcatOptions {
    fn default() -> Self {
        Self {
            join: Join::Exact,
            data_vars: Concat::Auto,
            coords: Concat::Minimal,
            compat: Compat::Override,
        }
    }
}

impl ConcatOptions {
    /// xarray's current (pre-2026) defaults: outer join, all data variables, coordinates
    /// that differ, `equals`.
    #[must_use]
    pub const fn legacy() -> Self {
        Self {
            join: Join::Outer,
            data_vars: Concat::All,
            coords: Concat::Different,
            compat: Compat::Equals,
        }
    }

    /// Set the join.
    #[must_use]
    pub const fn join(mut self, join: Join) -> Self {
        self.join = join;
        self
    }

    /// Set which data variables to concatenate.
    #[must_use]
    pub const fn data_vars(mut self, which: Concat) -> Self {
        self.data_vars = which;
        self
    }

    /// Set which coordinates to concatenate.
    #[must_use]
    pub const fn coords(mut self, which: Concat) -> Self {
        self.coords = which;
        self
    }

    /// Set the compatibility rule.
    #[must_use]
    pub const fn compat(mut self, compat: Compat) -> Self {
        self.compat = compat;
        self
    }
}

/// Options for [`merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MergeOptions {
    /// How to align dimensions.
    pub join: Join,
    /// How to combine variables of the same name.
    pub compat: Compat,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            join: Join::Exact,
            compat: Compat::Override,
        }
    }
}

impl MergeOptions {
    /// xarray's current (pre-2026) defaults: outer join, `no_conflicts`.
    #[must_use]
    pub const fn legacy() -> Self {
        Self {
            join: Join::Outer,
            compat: Compat::NoConflicts,
        }
    }

    /// Set the join.
    #[must_use]
    pub const fn join(mut self, join: Join) -> Self {
        self.join = join;
        self
    }

    /// Set the compatibility rule.
    #[must_use]
    pub const fn compat(mut self, compat: Compat) -> Self {
        self.compat = compat;
        self
    }
}

fn cannot(name: &Name, reason: impl Into<String>) -> CoreError {
    CoreError::CannotCombine {
        name: name.clone(),
        reason: reason.into(),
    }
}

fn is_numeric(d: DType) -> bool {
    !matches!(
        d,
        DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta | DType::Bool
    )
}

// ----- equality -----

/// xarray's `equals`: same dimensions, unit and values, with NaN equal to NaN.
fn equivalent<A: Repr, B: Repr>(a: &VariableBase<A>, b: &VariableBase<B>) -> Result<bool> {
    if a.dims() != b.dims()
        || a.shape() != b.shape()
        || a.unit() != b.unit()
        || a.time_zone() != b.time_zone()
    {
        return Ok(false);
    }
    if is_numeric(a.dtype()) && is_numeric(b.dtype()) {
        let (x, y) = (to_f64(&a.typed()?, None)?, to_f64(&b.typed()?, None)?);
        return Ok(x
            .iter()
            .zip(&y)
            .all(|(p, q)| p == q || (p.is_nan() && q.is_nan())));
    }
    Ok(variables_equal(a, b))
}

/// Combine variables of one name that are not concatenated.
fn combine_compat(parts: &[&Variable], compat: Compat, name: &Name) -> Result<Variable> {
    let Some((first, rest)) = parts.split_first() else {
        return Err(cannot(name, "no variables"));
    };
    match compat {
        Compat::Override => Ok((*first).clone()),
        Compat::Equals | Compat::Identical => {
            for v in rest {
                if !equivalent(first, v)? {
                    return Err(cannot(name, "values differ between objects"));
                }
                if compat == Compat::Identical && first.attrs() != v.attrs() {
                    return Err(cannot(name, "attributes differ between objects"));
                }
            }
            Ok((*first).clone())
        }
        Compat::NoConflicts => {
            let mut acc = (*first).clone();
            for v in rest {
                acc = fill_from(&acc, v, name)?;
            }
            Ok(acc)
        }
    }
}

/// `a` with its NaNs filled from `b`; an error where both have different non-NaN values.
fn fill_from(a: &Variable, b: &Variable, name: &Name) -> Result<Variable> {
    if equivalent(a, b)? {
        return Ok(a.clone());
    }
    if a.dims() != b.dims() || a.shape() != b.shape() || a.unit() != b.unit() {
        return Err(cannot(name, "dimensions or units differ between objects"));
    }
    if !(is_numeric(a.dtype()) && is_numeric(b.dtype())) {
        return Err(cannot(name, "values differ between objects"));
    }
    let (x, y) = (to_f64(&a.typed()?, None)?, to_f64(&b.typed()?, None)?);
    let mut out = Vec::with_capacity(x.len());
    for (p, q) in x.iter().zip(&y) {
        if !p.is_nan() && !q.is_nan() && p != q {
            return Err(cannot(name, "conflicting values"));
        }
        out.push(if p.is_nan() { *q } else { *p });
    }
    let data = ArrayD::from_shape_vec(IxDyn(a.shape()), out)
        .map_err(|_| cannot(name, "shape mismatch"))?;
    let mut v = Variable::from_typed(a.dims().iter().cloned(), TypedArray::F64(data))?;
    v.set_unit(a.unit().cloned());
    v.take_zone_from(a);
    v.attrs_mut()
        .extend(a.attrs().iter().map(|(k, x)| (k.clone(), x.clone())));
    Ok(v)
}

// ----- concatenation -----

/// `a` with an axis of length `len` inserted at `at` (if given) and axes permuted by `order`.
fn prep_as<T: Element>(
    a: &TypedArray<View<'_>>,
    insert: Option<(usize, usize)>,
    order: &[usize],
    _: Marker<T>,
) -> Option<TypedArray<Owned>> {
    let v = T::peek(a)?.view();
    let owned = match insert {
        Some((at, len)) => {
            let v = v.insert_axis(Axis(at));
            let mut shape = v.shape().to_vec();
            *shape.get_mut(at)? = len;
            v.broadcast(IxDyn(&shape))?.to_owned()
        }
        None => v.to_owned(),
    };
    if order.len() != owned.ndim() {
        return None;
    }
    Some(T::wrap(
        owned
            .permuted_axes(IxDyn(order))
            .as_standard_layout()
            .into_owned(),
    ))
}

fn concat_as<T: Element>(
    parts: &[TypedArray<Owned>],
    axis: usize,
    _: Marker<T>,
) -> Option<TypedArray<Owned>> {
    let views = parts
        .iter()
        .map(|p| T::peek(p).map(|a| a.view()))
        .collect::<Option<Vec<_>>>()?;
    ndarray::concatenate(Axis(axis), &views).ok().map(T::wrap)
}

/// Concatenate one variable's parts along `dim` (`lens`: each object's length along it).
fn concat_vars(parts: &[&Variable], dim: &Name, lens: &[usize], name: &Name) -> Result<Variable> {
    let bad = |why: &str| cannot(name, why);
    let template = parts
        .iter()
        .find(|v| v.dims().contains(dim))
        .or_else(|| parts.first())
        .ok_or_else(|| bad("no variables"))?;
    let mut target: Vec<Name> = template.dims().to_vec();
    if !target.contains(dim) {
        target.insert(0, dim.clone());
    }
    let axis = target
        .iter()
        .position(|d| d == dim)
        .ok_or_else(|| bad("internal: no axis"))?;
    let unit = template.unit();
    // Date-times keep one zone; joining zones would change how some of them read.
    if parts
        .iter()
        .any(|v| v.dtype() == DType::DateTime && v.time_zone() != template.time_zone())
    {
        return Err(bad(
            "date-times in different time zones; convert them to one zone first (tz_convert)",
        ));
    }

    // Lazy parts stay lazy when nothing needs converting or moving: same type and unit, and
    // either all share the target's dimensions or all gain the new dimension in front.
    let all_have = parts.iter().all(|v| v.dims() == target.as_slice());
    let all_new = parts
        .iter()
        .all(|v| !v.dims().contains(dim) && v.dims() == target.get(1..).unwrap_or_default());
    let uniform = parts
        .iter()
        .all(|v| v.dtype() == template.dtype() && v.unit() == unit);
    if parts.iter().any(|v| v.is_lazy()) && uniform && (all_have || all_new) {
        let sources = parts
            .iter()
            .map(|v| v.as_lazy_source())
            .collect::<Result<Vec<_>>>()?;
        let joined = crate::lazy::ConcatSource::new(sources, axis, all_new)?;
        let mut out =
            Variable::from_lazy(target, crate::LazyArray::new(std::sync::Arc::new(joined)))?;
        out.set_unit(unit.cloned());
        out.take_zone_from(template);
        out.attrs_mut()
            .extend(template.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        *out.encoding_mut() = template.encoding().clone();
        return Ok(out);
    }

    let mut arrays: Vec<TypedArray<Owned>> = Vec::with_capacity(parts.len());
    for (v, len) in parts.iter().zip(lens) {
        // Same unit as the template, converting if needed.
        let typed = match (v.unit(), unit) {
            (a, b) if a == b => v.typed()?.to_owned(),
            (Some(a), Some(b)) => {
                let conv = a.converter_to(b)?;
                TypedArray::F64(to_f64(&v.typed()?, Some(&conv))?)
            }
            _ => return Err(bad("a unit in one object and none in another")),
        };
        // Insert the dimension if missing, then order axes as the template.
        let mut dims: Vec<Name> = v.dims().to_vec();
        let insert = if dims.contains(dim) {
            None
        } else {
            dims.insert(0, dim.clone());
            Some((0, *len))
        };
        if dims.len() != target.len() || !dims.iter().all(|d| target.contains(d)) {
            return Err(bad("dimensions differ between objects"));
        }
        let order: Vec<usize> = target
            .iter()
            .filter_map(|t| dims.iter().position(|d| d == t))
            .collect();
        let view = typed.view();
        let prepped =
            crate::array::typed_dispatch!(&view, x => prep_as(&view, insert, &order, marker(x)));
        arrays.push(prepped.ok_or_else(|| bad("cannot broadcast"))?);
    }

    // A common element type.
    let first_dtype = arrays.first().map(TypedArray::dtype);
    if arrays.iter().any(|a| Some(a.dtype()) != first_dtype) {
        if !arrays.iter().all(|a| is_numeric(a.dtype())) {
            return Err(bad("element types differ and are not all numeric"));
        }
        arrays = arrays
            .iter()
            .map(|a| to_f64(&a.view(), None).map(TypedArray::F64))
            .collect::<Result<_>>()?;
    }
    let data = match arrays.first() {
        Some(f) => crate::array::typed_dispatch!(f, x => concat_as(&arrays, axis, marker(x))),
        None => None,
    }
    .ok_or_else(|| bad("shapes do not fit together"))?;

    let mut out = Variable::from_typed(target, data)?;
    out.set_unit(unit.cloned());
    out.take_zone_from(template);
    out.attrs_mut()
        .extend(template.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    *out.encoding_mut() = template.encoding().clone();
    Ok(out)
}

fn names<'a>(maps: impl Iterator<Item = &'a IndexMap<Name, Variable>>) -> Vec<Name> {
    let mut out: Vec<Name> = Vec::new();
    for m in maps {
        for k in m.keys() {
            if !out.contains(k) {
                out.push(k.clone());
            }
        }
    }
    out
}

/// Align `datasets` (owned copies; lazy data stays lazy when nothing moves).
fn aligned<S: Repr>(
    datasets: &[&DatasetBase<S>],
    join: Join,
    except: Option<&Name>,
) -> Result<Vec<Dataset>> {
    let mut objs: Vec<Obj<'_>> = datasets.iter().map(|d| Obj::of_dataset(*d)).collect();
    if let Some(d) = except {
        for o in &mut objs {
            o.indexes.shift_remove(d);
            o.sizes.shift_remove(d);
        }
    }
    let plan = plan(&objs, join)?;
    datasets
        .iter()
        .enumerate()
        .map(|(i, d)| apply_dataset(*d, &plan, i))
        .collect()
}

/// Concatenate datasets along `dim`, an existing dimension or a new one.
///
/// # Errors
/// Alignment errors ([`CoreError::NotAligned`] under the default exact join);
/// [`CoreError::CannotCombine`] for variables missing from some datasets, differing
/// dimensions, units or element types, or (with [`Compat::Equals`]) differing values;
/// unit errors for incompatible units; [`CoreError::Read`] if lazy data cannot be read.
pub fn concat_datasets<S: Repr>(
    datasets: &[&DatasetBase<S>],
    dim: &str,
    opts: &ConcatOptions,
) -> Result<Dataset> {
    let dim = Name::from(dim);
    let aligned = aligned(datasets, opts.join, Some(&dim))?;
    let Some(first) = aligned.first() else {
        return Err(cannot(&dim, "nothing to concatenate"));
    };
    let new_dim = aligned.iter().all(|d| !d.sizes().contains_key(&dim));
    let lens: Vec<usize> = aligned
        .iter()
        .map(|d| d.sizes().get(&dim).copied().unwrap_or(1))
        .collect();

    let mut out = Dataset::new();
    for is_coord in [true, false] {
        let maps: Vec<&IndexMap<Name, Variable>> = aligned
            .iter()
            .map(|d| if is_coord { d.coords() } else { d.data_vars() })
            .collect();
        let which = if is_coord {
            opts.coords
        } else {
            opts.data_vars
        };
        for name in names(maps.iter().copied()) {
            let parts: Vec<Option<&Variable>> = maps.iter().map(|m| m.get(&name)).collect();
            let present: Vec<&Variable> = parts.iter().flatten().copied().collect();
            let has_dim = present.iter().any(|v| v.dims().contains(&dim));
            let concat_it = has_dim
                || match which {
                    Concat::Auto => new_dim,
                    Concat::Minimal => false,
                    Concat::All => true,
                    Concat::Different => {
                        let mut differs = false;
                        for v in present.iter().skip(1) {
                            if let Some(f) = present.first() {
                                differs |= !equivalent(f, v)?;
                            }
                        }
                        differs || present.len() != parts.len()
                    }
                };
            let var = if concat_it {
                if present.len() != parts.len() {
                    return Err(cannot(&name, "present in some objects but not others"));
                }
                concat_vars(&present, &dim, &lens, &name)?
            } else {
                combine_compat(&present, opts.compat, &name)?
            };
            if is_coord {
                out.add_coord(name, var)?;
            } else {
                out.add_var(name, var)?;
            }
        }
    }
    out.attrs_mut()
        .extend(first.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    *out.encoding_mut() = first.encoding().clone();
    Ok(out)
}

const TMP: &str = "__annoray_data__";

/// Concatenate data arrays along `dim`, an existing dimension or a new one. The result has
/// the first array's name.
///
/// # Errors
/// As for [`concat_datasets`].
pub fn concat<S: Repr>(
    arrays: &[&DataArrayBase<S>],
    dim: &str,
    opts: &ConcatOptions,
) -> Result<DataArray> {
    let datasets = arrays
        .iter()
        .map(|a| {
            let mut ds = Dataset::new();
            ds.add_array(TMP, DataArrayBase::to_owned(*a))?;
            Ok(ds)
        })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Dataset> = datasets.iter().collect();
    let mut ds = concat_datasets(&refs, dim, &opts.data_vars(Concat::All))?;
    let var = ds.remove(TMP)?;
    let coords = ds
        .coords()
        .iter()
        .filter(|(_, c)| c.dims().iter().all(|d| var.dims().contains(d)))
        .map(|(k, c)| (k.clone(), c.clone()))
        .collect();
    let name = arrays.first().and_then(|a| a.name().cloned());
    Ok(DataArrayBase::from_parts(name, var, coords))
}

/// Merge datasets: align them, then combine variables of the same name. A dimension
/// coordinate (an index) takes precedence over non-index coordinates of the same name,
/// which are dropped, as in xarray.
///
/// # Errors
/// Alignment errors ([`CoreError::NotAligned`] under the default exact join);
/// [`CoreError::CannotCombine`] for conflicts under [`Compat::Equals`],
/// [`Compat::Identical`] or [`Compat::NoConflicts`], or a name that is a coordinate in one
/// dataset and a data variable in another.
pub fn merge<S: Repr>(datasets: &[&DatasetBase<S>], opts: &MergeOptions) -> Result<Dataset> {
    let aligned = aligned(datasets, opts.join, None)?;
    let mut out = Dataset::new();
    for is_coord in [true, false] {
        let maps: Vec<&IndexMap<Name, Variable>> = aligned
            .iter()
            .map(|d| if is_coord { d.coords() } else { d.data_vars() })
            .collect();
        for name in names(maps.iter().copied()) {
            if !is_coord && out.contains(&name) {
                return Err(cannot(
                    &name,
                    "a coordinate in one dataset, a variable in another",
                ));
            }
            let mut present: Vec<&Variable> = maps.iter().filter_map(|m| m.get(&name)).collect();
            // A dimension coordinate (an index) takes precedence over non-index ones of the
            // same name, which are dropped (xarray's merge rule).
            if is_coord {
                let index = |v: &&Variable| v.dims() == std::slice::from_ref(&name);
                if present.iter().any(index) && !present.iter().all(index) {
                    present.retain(index);
                }
            }
            let var = combine_compat(&present, opts.compat, &name)?;
            if is_coord {
                out.add_coord(name, var)?;
            } else {
                out.add_var(name, var)?;
            }
        }
    }
    if let Some(first) = aligned.first() {
        out.attrs_mut()
            .extend(first.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    }
    Ok(out)
}

/// The first label of a dimension coordinate, comparable across datasets.
fn first_key<S: Repr>(ds: &DatasetBase<S>, dim: &Name) -> Result<Vec<crate::align::Key>> {
    let c = ds
        .coords()
        .get(dim)
        .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
    Ok(match crate::lookup::index_keys(c, dim)? {
        crate::lookup::Keys::Num(v) => v.into_iter().map(crate::align::Key::F).collect(),
        crate::lookup::Keys::Str(v) => v.into_iter().map(crate::align::Key::S).collect(),
        crate::lookup::Keys::Ns(v, _) => v.into_iter().map(crate::align::Key::T).collect(),
    })
}

/// Combine datasets that tile one dimension (xarray's `combine_by_coords`, in one
/// dimension): find the dimension whose coordinates differ, order the datasets by it (its
/// coordinate may run either way, but must be monotonic once combined) and concatenate
/// along it. When no dimension differs, the datasets are merged. Lazy data stays lazy.
///
/// # Errors
/// [`CoreError::CannotCombine`] if the datasets differ along several dimensions, or the
/// combined coordinate is not monotonic; otherwise as for [`concat_datasets`] or [`merge`].
pub fn combine_by_coords<S: Repr>(
    datasets: &[&DatasetBase<S>],
    opts: &ConcatOptions,
) -> Result<Dataset> {
    let Some(first) = datasets.first() else {
        return Err(cannot(&Name::from(""), "nothing to combine"));
    };
    // Dimension coordinates every dataset has; those whose values differ.
    let mut varying = Vec::new();
    for (k, c) in first.coords() {
        if c.dims() != [k.clone()] || !datasets.iter().all(|d| d.coords().contains_key(k)) {
            continue;
        }
        let mut differs = false;
        for d in datasets.iter().skip(1) {
            if let Some(other) = d.coords().get(k) {
                differs |= !equivalent(c, other)?;
            }
        }
        if differs {
            varying.push(k.clone());
        }
    }
    let dim = match varying.as_slice() {
        [] => return merge(datasets, &MergeOptions::default()),
        [d] => d.clone(),
        many => {
            return Err(cannot(
                &Name::from(""),
                format!("datasets differ along several dimensions {many:?}; only one is supported"),
            ));
        }
    };
    let mut keyed = datasets
        .iter()
        .map(|d| first_key(d, &dim).map(|k| (k, *d)))
        .collect::<Result<Vec<_>>>()?;
    // The direction of the first dataset's coordinate decides the order.
    let descending = keyed
        .first()
        .and_then(|(k, _)| Some(k.get(1)? < k.first()?))
        .unwrap_or(false);
    keyed.sort_by(|(a, _), (b, _)| {
        let o = a.first().cmp(&b.first());
        if descending { o.reverse() } else { o }
    });
    let all: Vec<&crate::align::Key> = keyed.iter().flat_map(|(k, _)| k.iter()).collect();
    let monotonic = all.windows(2).all(|w| match w {
        [p, q] => {
            if descending {
                p > q
            } else {
                p < q
            }
        }
        _ => true,
    });
    if !monotonic {
        return Err(cannot(
            &dim,
            "the combined coordinate is not monotonic (overlapping or interleaved datasets)",
        ));
    }
    let ordered: Vec<&DatasetBase<S>> = keyed.into_iter().map(|(_, d)| d).collect();
    concat_datasets(&ordered, &dim, opts)
}
