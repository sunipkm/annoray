//! `stack` and `unstack`: several dimensions as one (xarray's multi-index dimensions).
//!
//! - `stack(new, dims)` moves `dims` to the end, in the given order, and flattens them into
//!   `new` (row-major). Every variable with any of `dims` is stacked (broadcast over those it
//!   lacks first). The coordinates of `dims` become *levels*: coordinates along `new`, each
//!   marked in its encoding (`level_of`) with `new` and its position; a dimension without a
//!   coordinate gets `0..n` as its level, as xarray.
//! - `unstack(dim)` reverses it: the levels' distinct values, in order of first appearance
//!   (xarray's `factorize`), become new dimensions at the end, in level order; every value
//!   goes to its combination. Combinations that are absent are filled with NaN (`fill_value`)
//!   for numbers, integers becoming `f64` unless the fill is a whole number they can hold, and
//!   NaT for times; strings and booleans need every combination. Repeated combinations are an
//!   error. Without level markers (a file, say), every coordinate along `dim` alone is a
//!   level, in coordinate order.
//! - Units, attributes and encodings are kept; no coordinate named after the stacked
//!   dimension is created (xarray's tuple-valued multi-index coordinate). Lazy data is read.

use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array::typed_map;
use crate::lookup::{Keys, index_keys};
use crate::{
    CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, NaT, Name, Repr, Result,
    TypedArray, Variable, VariableBase,
};

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

fn shape_err(e: ndarray::ShapeError) -> CoreError {
    invalid(e.to_string())
}

/// `var` with `dims` (broadcast to `sizes` where missing) flattened into `new` at the end.
fn stack_var<R: Repr>(
    var: &VariableBase<R>,
    dims: &[Name],
    new: &Name,
    sizes: &IndexMap<Name, usize>,
) -> Result<Variable> {
    if !dims.iter().any(|d| var.dims().contains(d)) {
        return Ok(var.to_owned());
    }
    let others: Vec<Name> = var
        .dims()
        .iter()
        .filter(|d| !dims.contains(d))
        .cloned()
        .collect();
    // Axes: the variable's, then length-1 axes for stacked dims it lacks.
    let missing: Vec<Name> = dims
        .iter()
        .filter(|d| !var.dims().contains(d))
        .cloned()
        .collect();
    let have: Vec<Name> = var.dims().iter().chain(&missing).cloned().collect();
    let perm: Vec<usize> = others
        .iter()
        .chain(dims)
        .map(|d| {
            have.iter()
                .position(|x| x == d)
                .ok_or_else(|| invalid("dim"))
        })
        .collect::<Result<_>>()?;
    let size = |d: &Name| {
        sizes
            .get(d)
            .copied()
            .ok_or_else(|| CoreError::NoSuchDim(d.clone()))
    };
    let full: Vec<usize> = others.iter().chain(dims).map(size).collect::<Result<_>>()?;
    let mut flat: Vec<usize> = others.iter().map(size).collect::<Result<_>>()?;
    let n = dims
        .iter()
        .map(size)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(CoreError::ArithmeticOverflow("stack"))?;
    flat.push(n);
    let extra = missing.len();
    let typed = var.typed()?.to_owned();
    let data = typed_map!(typed, a => {
        let mut a = a;
        for _ in 0..extra {
            let at = a.ndim();
            a = a.insert_axis(Axis(at));
        }
        let a = a.permuted_axes(IxDyn(&perm));
        let b = a
            .broadcast(IxDyn(&full))
            .ok_or_else(|| invalid("cannot broadcast for stacking"))?
            .as_standard_layout()
            .into_owned();
        b.into_shape_with_order(IxDyn(&flat)).map_err(shape_err)?
    });
    let mut out = Variable::from_typed(others.into_iter().chain([new.clone()]), data)?;
    out.set_unit(var.unit().cloned());
    out.attrs_mut()
        .extend(var.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    *out.encoding_mut() = var.encoding().clone();
    Ok(out)
}

/// The coordinates and variables of a stack: `(coords, stacked)`.
fn stack_parts<C: Repr>(
    coords: &IndexMap<Name, VariableBase<C>>,
    sizes: &IndexMap<Name, usize>,
    new: &Name,
    dims: &[Name],
) -> Result<IndexMap<Name, Variable>> {
    if sizes.contains_key(new) {
        return Err(CoreError::DuplicateDim(new.clone()));
    }
    for d in dims {
        if !sizes.contains_key(d) {
            return Err(CoreError::NoSuchDim(d.clone()));
        }
    }
    let mut out: IndexMap<Name, Variable> = IndexMap::new();
    for (k, c) in coords {
        let mut v = stack_var(c, dims, new, sizes)?;
        if let Some(i) = dims
            .iter()
            .position(|d| d == k && c.dims() == std::slice::from_ref(d))
        {
            v.encoding_mut().level_of = Some((new.clone(), i));
        }
        out.insert(k.clone(), v);
    }
    // Dimensions without a coordinate get 0..n as their level.
    for (i, d) in dims.iter().enumerate() {
        if !coords
            .get(d)
            .is_some_and(|c| c.dims() == std::slice::from_ref(d))
        {
            let n = sizes.get(d).copied().unwrap_or(0);
            let range: Vec<i64> = (0..n)
                .map(|j| i64::try_from(j).map_err(|_| CoreError::ArithmeticOverflow("stack")))
                .collect::<Result<_>>()?;
            let base = Variable::new(
                [d.clone()],
                ArrayD::from_shape_vec(IxDyn(&[n]), range).map_err(shape_err)?,
            )?;
            let mut v = stack_var(&base, dims, new, sizes)?;
            v.encoding_mut().level_of = Some((new.clone(), i));
            out.insert(d.clone(), v);
        }
    }
    Ok(out)
}

/// Codes (into first-appearance uniques) and the first position of each unique.
fn factorize<T: Eq + Hash + Clone>(v: &[T]) -> (Vec<usize>, Vec<usize>) {
    let mut seen: HashMap<T, usize> = HashMap::new();
    let mut firsts = Vec::new();
    let codes = v
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let next = firsts.len();
            *seen.entry(x.clone()).or_insert_with(|| {
                firsts.push(i);
                next
            })
        })
        .collect();
    (codes, firsts)
}

/// How a stacked dimension unstacks: level names, their coordinates, and where each
/// position goes in the product of the levels.
struct Plan {
    dim: Name,
    levels: Vec<(Name, Variable)>,
    sizes: Vec<usize>,
    target: Vec<usize>,
    complete: bool,
}

fn plan<C: Repr>(coords: &IndexMap<Name, VariableBase<C>>, dim: &Name, len: usize) -> Result<Plan> {
    let along = |c: &VariableBase<C>| c.dims() == std::slice::from_ref(dim);
    let mut marked: Vec<(usize, &Name, &VariableBase<C>)> = coords
        .iter()
        .filter_map(|(k, c)| match &c.encoding().level_of {
            Some((d, i)) if d == dim && along(c) => Some((*i, k, c)),
            _ => None,
        })
        .collect();
    marked.sort_by_key(|(i, _, _)| *i);
    let chosen: Vec<(&Name, &VariableBase<C>)> = if marked.is_empty() {
        coords
            .iter()
            .filter(|(k, c)| *k != dim && along(c))
            .collect()
    } else {
        marked.into_iter().map(|(_, k, c)| (k, c)).collect()
    };
    if chosen.is_empty() {
        return Err(invalid(format!("{dim:?} has no levels to unstack")));
    }
    let mut levels = Vec::new();
    let mut codes: Vec<Vec<usize>> = Vec::new();
    let mut sizes = Vec::new();
    for (k, c) in chosen {
        let (code, firsts) = match index_keys(c, dim)? {
            Keys::Num(v) => {
                if v.iter().any(|x| x.is_nan()) {
                    return Err(invalid(format!("level {k:?} has NaN")));
                }
                factorize(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            }
            Keys::Str(v) => factorize(&v),
            Keys::Ns(v, _) => factorize(&v),
        };
        let idx: Vec<isize> = firsts
            .iter()
            .map(|i| isize::try_from(*i).map_err(|_| CoreError::ArithmeticOverflow("unstack")))
            .collect::<Result<_>>()?;
        let mut level = c.take(dim, &idx)?.rename_dim(dim, k)?;
        level.encoding_mut().level_of = None;
        sizes.push(firsts.len());
        codes.push(code);
        levels.push((k.clone(), level));
    }
    let total = sizes
        .iter()
        .try_fold(1usize, |a, n| a.checked_mul(*n))
        .ok_or(CoreError::ArithmeticOverflow("unstack"))?;
    let mut target = Vec::with_capacity(len);
    let mut used = vec![false; total];
    for k in 0..len {
        let mut flat = 0usize;
        for (code, n) in codes.iter().zip(&sizes) {
            let c = code.get(k).copied().unwrap_or(0);
            flat = flat
                .checked_mul(*n)
                .and_then(|f| f.checked_add(c))
                .ok_or(CoreError::ArithmeticOverflow("unstack"))?;
        }
        match used.get_mut(flat) {
            Some(u) if *u => {
                return Err(invalid(format!(
                    "{dim:?} has repeated level combinations; cannot unstack"
                )));
            }
            Some(u) => *u = true,
            None => return Err(invalid("level code out of range")),
        }
        target.push(flat);
    }
    Ok(Plan {
        dim: dim.clone(),
        levels,
        sizes,
        complete: len == total,
        target,
    })
}

/// `var` with `plan.dim` split into the levels at the end (absent combinations filled).
fn unstack_var<R: Repr>(var: &VariableBase<R>, plan: &Plan, fill: Option<f64>) -> Result<Variable> {
    let Some(axis) = var.dims().iter().position(|d| *d == plan.dim) else {
        return Ok(var.to_owned());
    };
    let others: Vec<Name> = var
        .dims()
        .iter()
        .filter(|d| **d != plan.dim)
        .cloned()
        .collect();
    let mut perm: Vec<usize> = (0..var.ndim()).filter(|i| *i != axis).collect();
    perm.push(axis);
    let mut shape: Vec<usize> = var
        .shape()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, n)| *n)
        .collect();
    shape.extend(&plan.sizes);
    let last = others.len();
    let n = plan.target.len();
    let total: usize = plan.sizes.iter().product();
    // For each slot of the product: the source position, or `n` (the fill slot).
    let mut source = vec![n; total];
    for (k, t) in plan.target.iter().enumerate() {
        if let Some(s) = source.get_mut(*t) {
            *s = k;
        }
    }
    let mut typed = var.typed()?.to_owned();
    if !plan.complete {
        // A fill value the type can hold keeps the type; otherwise numbers become f64.
        let keeps = |d: DType| fill.is_some_and(|f| crate::elementwise::scalar_as(d, f).is_some());
        typed = match typed {
            TypedArray::F64(_) | TypedArray::F32(_) => typed,
            TypedArray::DateTime(_) | TypedArray::CfDate(_) | TypedArray::TimeDelta(_) => typed,
            t if matches!(t.dtype(), DType::Str | DType::Bool) => {
                return Err(CoreError::UnsupportedDType {
                    op: "unstack with absent combinations (no missing value for this type)",
                    dtype: t.dtype(),
                });
            }
            t if keeps(t.dtype()) => t,
            t => TypedArray::F64(crate::ops::to_f64(&t.view(), None)?),
        };
    }
    macro_rules! with_fill {
        ($a:expr, $fill:expr) => {{
            let a = $a
                .permuted_axes(IxDyn(&perm))
                .as_standard_layout()
                .into_owned();
            let a = if plan.complete {
                a
            } else {
                let mut fshape = a.shape().to_vec();
                if let Some(l) = fshape.last_mut() {
                    *l = 1;
                }
                let pad = ArrayD::from_elem(IxDyn(&fshape), $fill);
                ndarray::concatenate(Axis(last), &[a.view(), pad.view()]).map_err(shape_err)?
            };
            a.select(Axis(last), &source)
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order(IxDyn(&shape))
                .map_err(shape_err)?
        }};
    }
    let f = fill.unwrap_or(f64::NAN);
    #[allow(clippy::cast_possible_truncation)]
    let data = match typed {
        TypedArray::F64(a) => TypedArray::F64(with_fill!(a, f)),
        TypedArray::F32(a) => TypedArray::F32(with_fill!(a, f as f32)),
        TypedArray::DateTime(a) => {
            TypedArray::DateTime(with_fill!(a, <chrono::NaiveDateTime as NaT>::NAT))
        }
        TypedArray::CfDate(a) => {
            TypedArray::CfDate(with_fill!(a, <annoray_units::CfDate as NaT>::NAT))
        }
        TypedArray::TimeDelta(a) => {
            TypedArray::TimeDelta(with_fill!(a, <chrono::TimeDelta as NaT>::NAT))
        }
        other if plan.complete => typed_map!(other, a => {
            a.permuted_axes(IxDyn(&perm))
                .as_standard_layout()
                .into_owned()
                .select(Axis(last), &source)
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order(IxDyn(&shape))
                .map_err(shape_err)?
        }),
        other => {
            // Integers keeping their type: the fill is exactly representable.
            let Some(fill_t) = crate::elementwise::scalar_as(other.dtype(), f) else {
                return Err(invalid("fill value"));
            };
            match (other, fill_t) {
                (TypedArray::I8(a), TypedArray::I8(v)) => TypedArray::I8(with_fill!(a, first(&v)?)),
                (TypedArray::I16(a), TypedArray::I16(v)) => {
                    TypedArray::I16(with_fill!(a, first(&v)?))
                }
                (TypedArray::I32(a), TypedArray::I32(v)) => {
                    TypedArray::I32(with_fill!(a, first(&v)?))
                }
                (TypedArray::I64(a), TypedArray::I64(v)) => {
                    TypedArray::I64(with_fill!(a, first(&v)?))
                }
                (TypedArray::U8(a), TypedArray::U8(v)) => TypedArray::U8(with_fill!(a, first(&v)?)),
                (TypedArray::U16(a), TypedArray::U16(v)) => {
                    TypedArray::U16(with_fill!(a, first(&v)?))
                }
                (TypedArray::U32(a), TypedArray::U32(v)) => {
                    TypedArray::U32(with_fill!(a, first(&v)?))
                }
                (TypedArray::U64(a), TypedArray::U64(v)) => {
                    TypedArray::U64(with_fill!(a, first(&v)?))
                }
                _ => return Err(invalid("fill value")),
            }
        }
    };
    let mut out = Variable::from_typed(
        others
            .into_iter()
            .chain(plan.levels.iter().map(|(k, _)| k.clone())),
        data,
    )?;
    out.set_unit(var.unit().cloned());
    out.attrs_mut()
        .extend(var.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    let mut enc = var.encoding().clone();
    enc.level_of = None;
    *out.encoding_mut() = enc;
    Ok(out)
}

/// The single element of a 0-d fill array.
fn first<T: Clone>(a: &ArrayD<T>) -> Result<T> {
    a.iter()
        .next()
        .cloned()
        .ok_or_else(|| invalid("fill value"))
}

/// The coordinates after unstacking: levels become dimension coordinates, others along the
/// dimension are unstacked.
fn unstack_coords<C: Repr>(
    coords: &IndexMap<Name, VariableBase<C>>,
    plan: &Plan,
    fill: Option<f64>,
) -> Result<IndexMap<Name, Variable>> {
    let mut out = IndexMap::new();
    for (k, c) in coords {
        if plan.levels.iter().any(|(l, _)| l == k) {
            continue;
        }
        out.insert(k.clone(), unstack_var(c, plan, fill)?);
    }
    for (k, l) in &plan.levels {
        out.insert(k.clone(), l.clone());
    }
    Ok(out)
}

impl<S: Repr> DataArrayBase<S> {
    /// Flatten `dims` into a new dimension `new` at the end; see the module notes.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::DuplicateDim`] if `new` exists.
    pub fn stack(&self, new: &str, dims: &[&str]) -> Result<DataArray> {
        let new = Name::from(new);
        let dims: Vec<Name> = dims.iter().map(|d| Name::from(*d)).collect();
        let sizes: IndexMap<Name, usize> = self.sizes().map(|(d, n)| (d.clone(), n)).collect();
        let coords = stack_parts(self.coords(), &sizes, &new, &dims)?;
        let var = stack_var(self.variable(), &dims, &new, &sizes)?;
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Split the stacked dimension `dim` back into its levels; see the module notes.
    /// `fill_value` fills absent combinations (NaN/NaT by default).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] without levels, for repeated
    /// combinations or a NaN level; [`CoreError::UnsupportedDType`] for strings or booleans
    /// with absent combinations; [`CoreError::DuplicateDim`] if a level's name is a dimension.
    pub fn unstack(&self, dim: &str, fill_value: Option<f64>) -> Result<DataArray> {
        let dim = Name::from(dim);
        let len = self.variable().len_of(&dim)?;
        let plan = plan(self.coords(), &dim, len)?;
        for (k, _) in &plan.levels {
            if self.dims().contains(k) {
                return Err(CoreError::DuplicateDim(k.clone()));
            }
        }
        let coords = unstack_coords(self.coords(), &plan, fill_value)?;
        let var = unstack_var(self.variable(), &plan, fill_value)?;
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// The stacked dimensions (those with level markers).
    #[must_use]
    pub fn stacked_dims(&self) -> Vec<Name> {
        stacked(self.coords())
    }
}

fn stacked<C: Repr>(coords: &IndexMap<Name, VariableBase<C>>) -> Vec<Name> {
    let mut out: Vec<Name> = Vec::new();
    for c in coords.values() {
        if let Some((d, _)) = &c.encoding().level_of
            && !out.contains(d)
        {
            out.push(d.clone());
        }
    }
    out
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::stack`] of every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::stack`].
    pub fn stack(&self, new: &str, dims: &[&str]) -> Result<Dataset> {
        let new = Name::from(new);
        let dims: Vec<Name> = dims.iter().map(|d| Name::from(*d)).collect();
        let sizes = self.sizes();
        let coords = stack_parts(self.coords(), &sizes, &new, &dims)?;
        let mut out = Dataset::new();
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), stack_var(v, &dims, &new, &sizes)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// [`DataArrayBase::unstack`] of every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::unstack`].
    pub fn unstack(&self, dim: &str, fill_value: Option<f64>) -> Result<Dataset> {
        let dim = Name::from(dim);
        let len = self
            .sizes()
            .get(&dim)
            .copied()
            .ok_or_else(|| CoreError::NoSuchDim(dim.clone()))?;
        let plan = plan(self.coords(), &dim, len)?;
        let coords = unstack_coords(self.coords(), &plan, fill_value)?;
        let mut out = Dataset::new();
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), unstack_var(v, &plan, fill_value)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// The stacked dimensions (those with level markers).
    #[must_use]
    pub fn stacked_dims(&self) -> Vec<Name> {
        stacked(self.coords())
    }
}

// ----- files: CF compression by gathering -----

/// `ds` with each stacked dimension in CF "compression by gathering" form (CF §8.2, as
/// cf_xarray's `encode_multi_index_as_compress`): every level becomes a dimension coordinate
/// of its distinct values, and an integer coordinate named after the stacked dimension holds
/// each point's row-major position in the product of the levels, with `compress = "<levels>"`.
/// `None` when nothing is stacked. A stacked dimension that cannot be written this way
/// (repeated combinations, a name already taken) is left as it is.
pub(crate) fn compress_stacked<S: Repr>(ds: &DatasetBase<S>) -> Result<Option<Dataset>> {
    let dims = stacked(ds.coords());
    if dims.is_empty() {
        return Ok(None);
    }
    let sizes = ds.sizes();
    let mut drop: Vec<Name> = Vec::new();
    let mut add: Vec<(Name, Variable)> = Vec::new();
    for dim in dims {
        let Some(len) = sizes.get(&dim).copied() else {
            continue;
        };
        let Ok(p) = plan(ds.coords(), &dim, len) else {
            continue;
        };
        let taken = ds.coords().contains_key(&dim)
            || ds.data_vars().contains_key(&dim)
            || p.levels.iter().any(|(k, _)| sizes.contains_key(k));
        if taken {
            continue;
        }
        let codes: Vec<i64> = p
            .target
            .iter()
            .map(|t| i64::try_from(*t).map_err(|_| CoreError::ArithmeticOverflow("compress")))
            .collect::<Result<_>>()?;
        let mut index = Variable::new(
            [dim.clone()],
            ArrayD::from_shape_vec(IxDyn(&[codes.len()]), codes).map_err(shape_err)?,
        )?;
        let names: Vec<&str> = p.levels.iter().map(|(k, _)| k.as_str()).collect();
        index.attrs_mut().insert(
            Name::from("compress"),
            crate::AttrValue::Str(names.join(" ")),
        );
        for (k, level) in p.levels {
            drop.push(k.clone());
            add.push((k, level));
        }
        add.push((dim, index));
    }
    if add.is_empty() {
        return Ok(None);
    }
    let mut out = Dataset::new();
    for (k, c) in ds.coords() {
        if !drop.contains(k) {
            out.add_coord(k.clone(), c.to_owned())?;
        }
    }
    for (k, c) in add {
        out.add_coord(k, c)?;
    }
    for (k, v) in ds.data_vars() {
        out.add_var(k.clone(), v.to_owned())?;
    }
    *out.attrs_mut() = ds.attrs().clone();
    *out.encoding_mut() = ds.encoding().clone();
    Ok(Some(out))
}

/// The inverse of [`compress_stacked`] on a decoded dataset: an integer dimension coordinate
/// with a `compress` attribute naming dimension coordinates becomes marked levels along its
/// dimension. Indexes that do not fit (unknown names, out-of-range positions, levels also
/// used by other variables) are left as they are, with a warning.
pub(crate) fn decompress_stacked(ds: Dataset, warnings: &mut Vec<String>) -> Result<Dataset> {
    let indexes: Vec<(Name, Vec<Name>)> = ds
        .coords()
        .iter()
        .filter(|(k, c)| c.dims() == std::slice::from_ref(*k))
        .filter_map(|(k, c)| match c.attrs().get("compress") {
            Some(crate::AttrValue::Str(s)) => {
                Some((k.clone(), s.split_whitespace().map(Name::from).collect()))
            }
            _ => None,
        })
        .collect();
    if indexes.is_empty() {
        return Ok(ds);
    }
    let mut ds = ds;
    for (dim, names) in indexes {
        match levels_from_index(&ds, &dim, &names) {
            Ok(levels) => {
                let mut out = Dataset::new();
                for (k, c) in ds.coords() {
                    if *k != dim && !names.contains(k) {
                        out.add_coord(k.clone(), c.clone())?;
                    }
                }
                for (k, c) in levels {
                    out.add_coord(k, c)?;
                }
                for (k, v) in ds.data_vars() {
                    if !names.contains(k) {
                        out.add_var(k.clone(), v.clone())?;
                    }
                }
                *out.attrs_mut() = ds.attrs().clone();
                *out.encoding_mut() = ds.encoding().clone();
                ds = out;
            }
            Err(e) => warnings.push(format!("{dim}: compressed index left as is: {e}")),
        }
    }
    Ok(ds)
}

fn levels_from_index(ds: &Dataset, dim: &Name, names: &[Name]) -> Result<Vec<(Name, Variable)>> {
    let index = ds
        .coords()
        .get(dim)
        .ok_or_else(|| CoreError::NoSuchVariable(dim.clone()))?;
    let codes: Vec<f64> = index.to_f64()?.iter().copied().collect();
    let mut levels: Vec<&Variable> = Vec::new();
    for n in names {
        // A string level stored as characters decodes as a data variable.
        let c = ds
            .coords()
            .get(n)
            .or_else(|| ds.data_vars().get(n))
            .filter(|c| c.dims() == std::slice::from_ref(n))
            .ok_or_else(|| invalid(format!("no dimension coordinate {n:?}")))?;
        let used = ds
            .data_vars()
            .iter()
            .chain(ds.coords().iter())
            .any(|(k, v)| k != n && v.dims().contains(n));
        if used {
            return Err(invalid(format!(
                "{n:?} is also a dimension of other variables"
            )));
        }
        levels.push(c);
    }
    let sizes: Vec<usize> = levels
        .iter()
        .map(|c| c.shape().first().copied().unwrap_or(0))
        .collect();
    let total = sizes.iter().try_fold(1usize, |a, n| a.checked_mul(*n));
    // Row-major unravel of every code.
    let mut per_level: Vec<Vec<isize>> = vec![Vec::with_capacity(codes.len()); names.len()];
    for c in &codes {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mut rest = if c.fract() == 0.0 && *c >= 0.0 && total.is_some_and(|t| (*c as usize) < t)
        {
            *c as usize
        } else {
            return Err(invalid(format!(
                "position {c} is outside the levels' product"
            )));
        };
        for (slot, n) in per_level.iter_mut().zip(&sizes).rev() {
            let i = rest.checked_rem(*n).ok_or_else(|| invalid("empty level"))?;
            rest = rest.checked_div(*n).ok_or_else(|| invalid("empty level"))?;
            slot.push(isize::try_from(i).map_err(|_| CoreError::ArithmeticOverflow("unravel"))?);
        }
    }
    names
        .iter()
        .zip(levels)
        .zip(per_level)
        .enumerate()
        .map(|(i, ((n, c), idx))| {
            let mut v = c.take(n, &idx)?.rename_dim(n, dim)?;
            v.encoding_mut().level_of = Some((dim.clone(), i));
            Ok((n.clone(), v))
        })
        .collect()
}
