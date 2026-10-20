//! Shape and order operations (`squeeze`, `expand_dims`, `swap_dims`, `drop_sel`, `sortby`)
//! and sequential ones along a dimension (`diff`, `shift`, `roll`, `cumsum`, `cumprod`), as in
//! xarray. Each is built from selection, `take`, `where_` and the unit-checked arithmetic, so
//! lazy data stays lazy wherever those do (`cumsum`/`cumprod` read it).
//!
//! Deviations from xarray: `diff` with `n > 1` applies the label choice at every step
//! (xarray's recursion drops it after the first, so `label="lower"` mixes lower and upper
//! labels); `diff` of an absolute temperature gives a temperature difference (°C − °C is a
//! step in K), as the unit rules say.

use ndarray::{ArrayD, Axis, IxDyn};

use crate::lookup::{Keys, index_keys};
use crate::{
    BinOp, CoreError, DataArray, DataArrayBase, Dataset, DatasetBase, Indexer, IntoLabelIndexer,
    Name, PySlice, Repr, Result, TypedArray, Variable, VariableBase,
};

fn slice(start: Option<isize>, stop: Option<isize>) -> Indexer {
    Indexer::Slice(PySlice::new(start, stop, None))
}

/// Positions `0..n` of `dim`, as `take` indices.
fn positions(n: usize) -> Result<Vec<isize>> {
    (0..n)
        .map(|i| isize::try_from(i).map_err(|_| CoreError::ArithmeticOverflow("index")))
        .collect()
}

/// The order that sorts `keys` ascending (or descending), NaN/NaT last, ties kept in order.
fn sort_order(keys: &Keys, ascending: bool) -> Vec<usize> {
    let n = match keys {
        Keys::Num(v) => v.len(),
        Keys::Str(v) => v.len(),
        Keys::Ns(v, _) => v.len(),
    };
    let mut order: Vec<usize> = (0..n).collect();
    let flip = |o: std::cmp::Ordering| if ascending { o } else { o.reverse() };
    match keys {
        Keys::Num(v) => order.sort_by(|a, b| {
            let (p, q) = (
                v.get(*a).copied().unwrap_or(f64::NAN),
                v.get(*b).copied().unwrap_or(f64::NAN),
            );
            match (p.is_nan(), q.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => flip(p.total_cmp(&q)),
            }
        }),
        Keys::Str(v) => order.sort_by(|a, b| flip(v.get(*a).cmp(&v.get(*b)))),
        Keys::Ns(v, _) => order.sort_by(|a, b| {
            let (p, q) = (
                v.get(*a).copied().unwrap_or(i128::MIN),
                v.get(*b).copied().unwrap_or(i128::MIN),
            );
            match (p == i128::MIN, q == i128::MIN) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => flip(p.cmp(&q)),
            }
        }),
    }
    order
}

impl<S: Repr> DataArrayBase<S> {
    /// Drop length-1 dimensions (all of them for an empty `dims`), keeping their coordinates
    /// as scalar coordinates.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] if a named dimension is not
    /// of length 1.
    pub fn squeeze(&self, dims: &[&str]) -> Result<DataArray> {
        let chosen: Vec<Name> = if dims.is_empty() {
            self.dims()
                .iter()
                .zip(self.shape())
                .filter(|(_, n)| **n == 1)
                .map(|(d, _)| d.clone())
                .collect()
        } else {
            for d in dims {
                if self.variable().len_of(d)? != 1 {
                    return Err(CoreError::InvalidReduction(format!(
                        "cannot squeeze dimension {d:?}: its length is not 1"
                    )));
                }
            }
            dims.iter().map(|d| Name::from(*d)).collect()
        };
        Ok(self
            .isel(chosen.iter().map(|d| (d.as_str(), 0isize)))?
            .to_owned())
    }

    /// Insert a new dimension of length 1 at position `axis` (counted from the end when
    /// negative, so `-1` appends). A scalar coordinate of that name becomes its coordinate.
    ///
    /// # Errors
    /// [`CoreError::DuplicateDim`] if the dimension exists; [`CoreError::InvalidReduction`]
    /// for an axis out of range.
    pub fn expand_dims(&self, dim: &str, axis: isize) -> Result<DataArray> {
        let var = self.variable().expand_dims(dim, axis)?;
        let name = Name::from(dim);
        let mut coords = indexmap::IndexMap::new();
        for (k, c) in self.coords() {
            if *k == name && c.ndim() == 0 {
                coords.insert(k.clone(), c.expand_dims(dim, 0)?);
            } else {
                coords.insert(k.clone(), c.to_owned());
            }
        }
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Make coordinate `new` (one-dimensional along `old`) the index of that dimension,
    /// renaming the dimension to `new` everywhere; the old dimension coordinate stays, as a
    /// coordinate along `new`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] if `new` is not a coordinate
    /// along `old` alone.
    pub fn swap_dims(&self, old: &str, new: &str) -> Result<DataArray> {
        check_swap(self.coords().get(new), old, new)?;
        self.variable().axis(old)?;
        let to = Name::from(new);
        let var = self.variable().to_owned().rename_dim(old, &to)?;
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| Ok((k.clone(), c.to_owned().rename_dim(old, &to)?)))
            .collect::<Result<_>>()?;
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Rename dimensions and coordinates (`old → new` pairs; xarray's `rename` with a
    /// mapping): a name may be a dimension, a coordinate, or both.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] for a name that is neither; [`CoreError::DuplicateDim`] or
    /// [`CoreError::NameConflict`] if a new name is taken.
    pub fn rename_dims(&self, pairs: &[(&str, &str)]) -> Result<DataArray> {
        let mut var = self.variable().to_owned();
        let mut coords: indexmap::IndexMap<Name, Variable> = self
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        for (old, new) in pairs {
            let to = Name::from(*new);
            let is_dim = var.dims().iter().any(|d| d == *old)
                || coords.values().any(|c| c.dims().iter().any(|d| d == *old));
            let is_coord = coords.contains_key(*old);
            if !is_dim && !is_coord {
                return Err(CoreError::NoSuchDim(Name::from(*old)));
            }
            if is_dim {
                if var.dims().contains(&to) {
                    return Err(CoreError::DuplicateDim(to));
                }
                if var.dims().iter().any(|d| d == *old) {
                    var = var.rename_dim(old, &to)?;
                }
                coords = coords
                    .into_iter()
                    .map(|(k, c)| {
                        let c = if c.dims().iter().any(|d| d == *old) {
                            c.rename_dim(old, &to)?
                        } else {
                            c
                        };
                        Ok((k, c))
                    })
                    .collect::<Result<_>>()?;
            }
            if is_coord {
                if coords.contains_key(&to) {
                    return Err(CoreError::NameConflict(to));
                }
                coords = coords
                    .into_iter()
                    .map(|(k, c)| if k == *old { (to.clone(), c) } else { (k, c) })
                    .collect();
            }
        }
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Drop the entries of `dim` whose labels are `labels` (each must exist).
    ///
    /// # Errors
    /// [`CoreError::NoIndex`], [`CoreError::LabelNotFound`].
    pub fn drop_sel<L: IntoLabelIndexer>(
        &self,
        dim: &str,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<DataArray> {
        let keep = kept(self.coords(), self.variable().len_of(dim)?, dim, labels)?;
        self.take(dim, &keep)
    }

    /// Reorder `dim` by the values of the coordinate `by` (one-dimensional along `dim`; the
    /// dimension's own coordinate when `by` is `dim`). NaN and NaT go last; ties keep their
    /// order.
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if there is no such coordinate along `dim`.
    pub fn sortby(&self, dim: &str, by: &str, ascending: bool) -> Result<DataArray> {
        let order = sort_positions(self.coords().get(by), dim, by, ascending)?;
        self.take(dim, &order)
    }

    /// The `n`-th discrete difference along `dim` (`x[i+1] − x[i]`, `n` times), labelled with
    /// the upper coordinates (or the lower ones with `upper = false`). Units follow
    /// subtraction: an absolute temperature gives a temperature difference.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; unit and dtype errors of subtraction (strings, booleans);
    /// [`CoreError::ArithmeticOverflow`] for integers.
    pub fn diff(&self, dim: &str, n: usize, upper: bool) -> Result<DataArray> {
        let mut out = self.to_owned();
        for _ in 0..n {
            out = diff_once(&out, dim, upper)?;
        }
        Ok(out)
    }

    /// Shift the values by `n` positions along `dim` (positive: towards higher indices),
    /// filling with NaN (NaT for times); coordinates stay. Integers become `f64`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::UnsupportedDType`] for types without a
    /// missing value (strings, booleans).
    pub fn shift(&self, dim: &str, n: isize) -> Result<DataArray> {
        let len = self.variable().len_of(dim)?;
        let len_i = isize::try_from(len).map_err(|_| CoreError::ArithmeticOverflow("shift"))?;
        let last = len_i.saturating_sub(1).max(0);
        let source: Vec<isize> = (0..len_i)
            .map(|i| i.saturating_sub(n).clamp(0, last))
            .collect();
        let valid: Vec<bool> = (0..len_i)
            .map(|i| {
                let j = i.saturating_sub(n);
                j >= 0 && j < len_i
            })
            .collect();
        let taken = self.take(dim, &source)?;
        let taken = DataArrayBase::from_parts(
            taken.name().cloned(),
            taken.variable().to_owned(),
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        );
        let cond = DataArray::from_array(
            [dim],
            ArrayD::from_shape_vec(IxDyn(&[len]), valid)
                .map_err(|e| CoreError::InvalidReduction(e.to_string()))?,
        )?;
        taken.where_(&cond)
    }

    /// Rotate the values by `n` positions along `dim` (positive: towards higher indices).
    /// Coordinates along `dim` rotate too only with `roll_coords`, as in xarray.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`].
    pub fn roll(&self, dim: &str, n: isize, roll_coords: bool) -> Result<DataArray> {
        let len = self.variable().len_of(dim)?;
        let len_i = isize::try_from(len).map_err(|_| CoreError::ArithmeticOverflow("roll"))?;
        let order: Vec<isize> = (0..len_i)
            .map(|i| i.saturating_sub(n).rem_euclid(len_i.max(1)))
            .collect();
        let rolled = self.take(dim, &order)?;
        if roll_coords {
            return Ok(rolled);
        }
        Ok(DataArrayBase::from_parts(
            rolled.name().cloned(),
            rolled.variable().to_owned(),
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }

    /// Running sum along `dim`, skipping NaN (a NaN position carries the running total, as
    /// xarray's default). Integers stay integers (checked); booleans count as integers.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::UnsupportedDType`] for strings and times; unit
    /// errors for absolute temperatures; [`CoreError::ArithmeticOverflow`].
    pub fn cumsum(&self, dim: &str) -> Result<DataArray> {
        if let Some(u) = self.unit()
            && !matches!(u.kind(), annoray_units::UnitKind::Linear)
        {
            return Err(annoray_units::UnitError::UnsupportedOp {
                op: "a cumulative sum",
                unit: u.to_string(),
            }
            .into());
        }
        let var = cumulative(self.variable(), dim, Cumulative::Sum)?;
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            var,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }

    /// Running product along `dim`, skipping NaN. Only dimensionless data (each product has
    /// another unit otherwise).
    ///
    /// # Errors
    /// As for [`cumsum`](Self::cumsum); [`UnitError::UnsupportedOp`](annoray_units::UnitError)
    /// for data with a unit.
    pub fn cumprod(&self, dim: &str) -> Result<DataArray> {
        if let Some(u) = self.unit()
            && !u.is_dimensionless()
        {
            return Err(annoray_units::UnitError::UnsupportedOp {
                op: "a cumulative product (its unit would vary along the dimension)",
                unit: u.to_string(),
            }
            .into());
        }
        let var = cumulative(self.variable(), dim, Cumulative::Prod)?;
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            var,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }
}

fn check_swap<R: Repr>(coord: Option<&VariableBase<R>>, old: &str, new: &str) -> Result<()> {
    match coord {
        Some(c) if c.dims().len() == 1 && c.dims().first().is_some_and(|d| d == old) => Ok(()),
        _ => Err(CoreError::InvalidReduction(format!(
            "swap_dims: {new:?} must be a coordinate along {old:?} only"
        ))),
    }
}

fn kept<R: Repr, L: IntoLabelIndexer>(
    coords: &indexmap::IndexMap<Name, VariableBase<R>>,
    len: usize,
    dim: &str,
    labels: impl IntoIterator<Item = L>,
) -> Result<Vec<isize>> {
    let labels = labels
        .into_iter()
        .map(IntoLabelIndexer::into_label_indexer)
        .collect::<Result<Vec<_>>>()?;
    let dropped = crate::select::label_positions(coords, dim, labels)?;
    Ok(positions(len)?
        .into_iter()
        .filter(|i| !dropped.contains(i))
        .collect())
}

fn sort_positions<R: Repr>(
    by: Option<&VariableBase<R>>,
    dim: &str,
    name: &str,
    ascending: bool,
) -> Result<Vec<isize>> {
    let coord = by
        .filter(|c| c.dims().len() == 1 && c.dims().first().is_some_and(|d| d == dim))
        .ok_or_else(|| CoreError::NoIndex(Name::from(name)))?;
    let keys = index_keys(coord, &Name::from(dim))?;
    sort_order(&keys, ascending)
        .into_iter()
        .map(|i| isize::try_from(i).map_err(|_| CoreError::ArithmeticOverflow("sortby")))
        .collect()
}

fn diff_once<S: Repr>(da: &DataArrayBase<S>, dim: &str, upper: bool) -> Result<DataArray> {
    let hi = da.isel([(dim, slice(Some(1), None))])?;
    let lo = da.isel([(dim, slice(None, Some(-1)))])?;
    // Positional subtraction: the operands without their coordinates along `dim`.
    let strip = |a: &DataArrayBase<crate::View<'_>>| {
        DataArrayBase::from_parts(
            a.name().cloned(),
            a.variable().to_owned(),
            a.coords()
                .iter()
                .filter(|(_, c)| !c.dims().iter().any(|d| d == dim))
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        )
    };
    let values = strip(&hi).binary(
        BinOp::Sub,
        &strip(&lo),
        annoray_units::UnitPolicy::default(),
    )?;
    let labels = if upper { &hi } else { &lo };
    let mut coords: indexmap::IndexMap<Name, Variable> = labels
        .coords()
        .iter()
        .filter(|(_, c)| c.dims().iter().any(|d| d == dim))
        .map(|(k, c)| (k.clone(), c.to_owned()))
        .collect();
    for (k, c) in values.coords() {
        coords.entry(k.clone()).or_insert_with(|| c.clone());
    }
    let ordered = da
        .coords()
        .keys()
        .filter_map(|k| coords.get(k).map(|c| (k.clone(), c.clone())))
        .collect();
    Ok(DataArrayBase::from_parts(
        da.name().cloned(),
        values.variable().to_owned(),
        ordered,
    ))
}

#[derive(Clone, Copy)]
enum Cumulative {
    Sum,
    Prod,
}

fn cumulative<S: Repr>(v: &VariableBase<S>, dim: &str, op: Cumulative) -> Result<Variable> {
    let axis = Axis(v.axis(dim)?);
    let typed = v.typed()?;
    macro_rules! float {
        ($a:expr, $t:ty) => {{
            let mut out = $a.to_owned();
            let (unit, step): ($t, fn($t, $t) -> $t) = match op {
                Cumulative::Sum => (0.0, |acc, x| acc + x),
                Cumulative::Prod => (1.0, |acc, x| acc * x),
            };
            for mut lane in out.lanes_mut(axis) {
                let mut acc = unit;
                for x in &mut lane {
                    if !x.is_nan() {
                        acc = step(acc, *x);
                    }
                    *x = acc;
                }
            }
            out
        }};
    }
    macro_rules! int {
        ($a:expr) => {{
            let mut out = $a.mapv(i64::from);
            for mut lane in out.lanes_mut(axis) {
                let mut acc: i64 = match op {
                    Cumulative::Sum => 0,
                    Cumulative::Prod => 1,
                };
                for x in &mut lane {
                    acc = match op {
                        Cumulative::Sum => acc.checked_add(*x),
                        Cumulative::Prod => acc.checked_mul(*x),
                    }
                    .ok_or(CoreError::ArithmeticOverflow("cumulative"))?;
                    *x = acc;
                }
            }
            TypedArray::I64(out)
        }};
    }
    let data = match &typed {
        TypedArray::F64(a) => TypedArray::F64(float!(a, f64)),
        TypedArray::F32(a) => TypedArray::F32(float!(a, f32)),
        TypedArray::I8(a) => int!(a),
        TypedArray::I16(a) => int!(a),
        TypedArray::I32(a) => int!(a),
        TypedArray::I64(a) => int!(a),
        TypedArray::U8(a) => int!(a),
        TypedArray::U16(a) => int!(a),
        TypedArray::U32(a) => int!(a),
        TypedArray::Bool(a) => int!(a.mapv(i64::from)),
        other => {
            return Err(CoreError::UnsupportedDType {
                op: "cumulative",
                dtype: other.dtype(),
            });
        }
    };
    let mut out = Variable::from_typed(v.dims().iter().cloned(), data)?;
    out.set_unit(v.unit().cloned());
    out.take_zone_from(v);
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

impl<S: Repr> VariableBase<S> {
    /// The variable with a new length-1 dimension at `axis` (negative counts from the end).
    ///
    /// # Errors
    /// [`CoreError::DuplicateDim`]; [`CoreError::InvalidReduction`] for an axis out of range.
    pub fn expand_dims(&self, dim: &str, axis: isize) -> Result<Variable> {
        let name = Name::from(dim);
        if self.dims().contains(&name) {
            return Err(CoreError::DuplicateDim(name));
        }
        let rank =
            isize::try_from(self.ndim()).map_err(|_| CoreError::ArithmeticOverflow("rank"))?;
        let at = if axis < 0 {
            rank.saturating_add(1).saturating_add(axis)
        } else {
            axis
        };
        let at = usize::try_from(at)
            .ok()
            .filter(|a| *a <= self.ndim())
            .ok_or_else(|| CoreError::InvalidReduction(format!("axis {axis} is out of range")))?;
        let mut dims: Vec<Name> = self.dims().to_vec();
        dims.insert(at, name);
        let mut out = if self.is_lazy() {
            let src = crate::lazy::InsertAxisSource::new(self.as_lazy_source()?, at);
            Variable::from_lazy(dims, crate::LazyArray::new(std::sync::Arc::new(src)))?
        } else {
            Variable::from_typed(dims, self.typed()?.to_owned().insert_axis_owned(at))?
        };
        out.set_unit(self.unit().cloned());
        out.take_zone_from(self);
        *out.encoding_mut() = self.encoding().clone();
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }
}

/// Apply a DataArray operation to every data variable of a dataset that has `dim`, keeping
/// the others; coordinates come from the results (or are kept).
pub(crate) fn per_variable<S: Repr>(
    ds: &DatasetBase<S>,
    dim: &str,
    f: impl Fn(&DataArrayBase<crate::View<'_>>) -> Result<DataArray>,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    let mut coords: indexmap::IndexMap<Name, Variable> = indexmap::IndexMap::new();
    let mut vars = Vec::new();
    for k in ds.data_vars().keys() {
        let da = ds.get(k)?;
        if da.dims().iter().any(|d| d == dim) {
            let r = f(&da)?;
            for (ck, c) in r.coords() {
                coords.entry(ck.clone()).or_insert_with(|| c.clone());
            }
            vars.push((k.clone(), r.variable().to_owned()));
        } else {
            vars.push((k.clone(), da.variable().to_owned()));
        }
    }
    for (k, c) in ds.coords() {
        let c = coords.shift_remove(k).unwrap_or_else(|| c.to_owned());
        out.add_coord(k.clone(), c)?;
    }
    for (k, c) in coords {
        out.add_coord(k, c)?;
    }
    for (k, v) in vars {
        out.add_var(k, v)?;
    }
    *out.attrs_mut() = ds.attrs().clone();
    Ok(out)
}

impl<S: Repr> DatasetBase<S> {
    /// Drop length-1 dimensions (all of them for an empty `dims`), keeping their coordinates
    /// as scalars.
    ///
    /// # Errors
    /// As for [`DataArrayBase::squeeze`].
    pub fn squeeze(&self, dims: &[&str]) -> Result<Dataset> {
        let sizes = self.sizes();
        let chosen: Vec<String> = if dims.is_empty() {
            sizes
                .iter()
                .filter(|(_, n)| **n == 1)
                .map(|(d, _)| d.to_string())
                .collect()
        } else {
            for d in dims {
                match sizes.get(*d) {
                    Some(1) => {}
                    Some(_) => {
                        return Err(CoreError::InvalidReduction(format!(
                            "cannot squeeze dimension {d:?}: its length is not 1"
                        )));
                    }
                    None => return Err(CoreError::NoSuchDim(Name::from(*d))),
                }
            }
            dims.iter().map(ToString::to_string).collect()
        };
        Ok(self
            .isel(chosen.iter().map(|d| (d.as_str(), 0isize)))?
            .to_owned())
    }

    /// Make coordinate `new` (along `old`) the index of that dimension; see
    /// [`DataArrayBase::swap_dims`].
    ///
    /// # Errors
    /// As for [`DataArrayBase::swap_dims`].
    pub fn swap_dims(&self, old: &str, new: &str) -> Result<Dataset> {
        check_swap(self.coords().get(new), old, new)?;
        let to = Name::from(new);
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(k.clone(), c.to_owned().rename_dim(old, &to)?)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), v.to_owned().rename_dim(old, &to)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// Drop the entries of `dim` labelled `labels` from every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::drop_sel`].
    pub fn drop_sel<L: IntoLabelIndexer>(
        &self,
        dim: &str,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<Dataset> {
        let len = self
            .sizes()
            .get(dim)
            .copied()
            .ok_or_else(|| CoreError::NoSuchDim(Name::from(dim)))?;
        let keep = kept(self.coords(), len, dim, labels)?;
        self.take(dim, &keep)
    }

    /// Reorder `dim` by the coordinate `by`; see [`DataArrayBase::sortby`].
    ///
    /// # Errors
    /// As for [`DataArrayBase::sortby`].
    pub fn sortby(&self, dim: &str, by: &str, ascending: bool) -> Result<Dataset> {
        let order = sort_positions(self.coords().get(by), dim, by, ascending)?;
        self.take(dim, &order)
    }

    /// [`DataArrayBase::diff`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::diff`].
    pub fn diff(&self, dim: &str, n: usize, upper: bool) -> Result<Dataset> {
        per_variable(self, dim, |da| da.diff(dim, n, upper))
    }

    /// [`DataArrayBase::shift`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::shift`].
    pub fn shift(&self, dim: &str, n: isize) -> Result<Dataset> {
        per_variable(self, dim, |da| da.shift(dim, n))
    }

    /// [`DataArrayBase::roll`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::roll`].
    pub fn roll(&self, dim: &str, n: isize, roll_coords: bool) -> Result<Dataset> {
        per_variable(self, dim, |da| da.roll(dim, n, roll_coords))
    }

    /// [`DataArrayBase::cumsum`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::cumsum`].
    pub fn cumsum(&self, dim: &str) -> Result<Dataset> {
        per_variable(self, dim, |da| da.cumsum(dim))
    }

    /// [`DataArrayBase::cumprod`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::cumprod`].
    pub fn cumprod(&self, dim: &str) -> Result<Dataset> {
        per_variable(self, dim, |da| da.cumprod(dim))
    }
}
