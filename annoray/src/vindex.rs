//! Vectorized (pointwise) indexing, as xarray's `isel`/`sel` with `DataArray` indexers.
//!
//! - Each indexer is an integer array with its own dimensions (or a 1-D boolean mask,
//!   taken as its true positions). Indexers broadcast against each other by name; the
//!   result picks one element per point of that broadcast shape, so
//!   `isel_vectorized([("x", ix), ("y", iy)])` with `ix`, `iy` along `points` gives the
//!   values at `(ix[p], iy[p])` along `points`.
//! - Result dimensions follow the array's order, each indexed dimension replaced (at its
//!   position) by its indexer's dimensions not already placed. A dimension of the array that
//!   is not indexed but that an indexer also has is matched elementwise (xarray turns it
//!   into `arange`): `isel_vectorized([("x", i)])` with `i` along the array's `y` picks
//!   `(i[j], j)` along `y`.
//! - Coordinates along indexed dimensions are indexed alike; the indexers' own coordinates
//!   are attached (one that disagrees with an existing coordinate of the same name is an
//!   error, except, for `sel`, the indexed dimensions' coordinates, which the selection
//!   determines).
//! - Lazy data is read only within the bounding box of the selected positions.
//! - `sel_vectorized` turns label arrays (numbers, in the coordinate's unit or as
//!   quantities; strings; dates) into positions with the usual `Method` and tolerance.

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array::typed_map;
use crate::ops::broadcast_to_dims;
use crate::{
    CoreError, DataArray, DataArrayBase, Dataset, DatasetBase, IndexError, Indexer, Label,
    LabelIndexer, Method, Name, PySlice, Repr, Result, TypedArray, Variable, VariableBase,
    normalize_index, variables_equal,
};

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// Positions (normalized) along one dimension, with the indexer's dimensions.
#[derive(Clone)]
struct Ix {
    dims: Vec<Name>,
    pos: ArrayD<usize>,
}

/// An indexer for `dim` (of length `len`) from an integer array or a 1-D boolean mask.
fn ix_from<R: Repr>(dim: &Name, len: usize, idx: &VariableBase<R>) -> Result<Ix> {
    let typed = idx.typed()?;
    let ints: ArrayD<i64> = match &typed {
        TypedArray::Bool(m) => {
            if m.ndim() != 1 {
                return Err(invalid("boolean indexers must be one-dimensional"));
            }
            let hits: Vec<i64> = m
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i64::try_from(i).map_err(|_| IndexError::Overflow))
                .collect::<std::result::Result<_, _>>()?;
            ArrayD::from_shape_vec(IxDyn(&[hits.len()]), hits)
                .map_err(|e| invalid(e.to_string()))?
        }
        TypedArray::I8(a) => a.mapv(i64::from),
        TypedArray::I16(a) => a.mapv(i64::from),
        TypedArray::I32(a) => a.mapv(i64::from),
        TypedArray::I64(a) => a.to_owned(),
        TypedArray::U8(a) => a.mapv(i64::from),
        TypedArray::U16(a) => a.mapv(i64::from),
        TypedArray::U32(a) => a.mapv(i64::from),
        TypedArray::U64(a) => a.mapv(|v| i64::try_from(v).unwrap_or(i64::MAX)),
        other => {
            return Err(CoreError::UnsupportedDType {
                op: "vectorized indexing (positions must be integers)",
                dtype: other.dtype(),
            });
        }
    };
    let mut pos = ArrayD::from_elem(IxDyn(ints.shape()), 0usize);
    for (p, i) in pos.iter_mut().zip(&ints) {
        let i = isize::try_from(*i).map_err(|_| IndexError::Overflow)?;
        *p = normalize_index(i, len, dim)?;
    }
    Ok(Ix {
        dims: idx.dims().to_vec(),
        pos,
    })
}

/// `var` indexed by those of `ix` along its dimensions; unchanged without any.
fn vindex_var<R: Repr>(var: &VariableBase<R>, ix: &IndexMap<Name, Ix>) -> Result<Variable> {
    let mut mine: IndexMap<Name, Ix> = var
        .dims()
        .iter()
        .filter_map(|d| ix.get(d).map(|i| (d.clone(), i.clone())))
        .collect();
    if mine.is_empty() {
        return Ok(var.to_owned());
    }
    // A free dimension that an indexer also has is matched elementwise (arange).
    let idx_dims: Vec<Name> = mine.values().flat_map(|i| i.dims.clone()).collect();
    for (d, n) in var.dims().iter().zip(var.shape()) {
        if !mine.contains_key(d) && idx_dims.contains(d) {
            let pos = ArrayD::from_shape_vec(IxDyn(&[*n]), (0..*n).collect())
                .map_err(|e| invalid(e.to_string()))?;
            mine.insert(
                d.clone(),
                Ix {
                    dims: vec![d.clone()],
                    pos,
                },
            );
        }
    }
    // Output dimensions, and the broadcast (point) dimensions with their sizes.
    let mut out_dims: Vec<Name> = Vec::new();
    let mut point_sizes: IndexMap<Name, usize> = IndexMap::new();
    for d in var.dims() {
        match mine.get(d) {
            None => out_dims.push(d.clone()),
            Some(i) => {
                for (pd, n) in i.dims.iter().zip(i.pos.shape()) {
                    match point_sizes.get(pd) {
                        Some(m) if m != n => {
                            return Err(CoreError::DimLengthMismatch {
                                dim: pd.clone(),
                                expected: *m,
                                found: *n,
                            });
                        }
                        Some(_) => {}
                        None => {
                            point_sizes.insert(pd.clone(), *n);
                            out_dims.push(pd.clone());
                        }
                    }
                }
            }
        }
    }
    // Other indexers may add point dimensions only through dims already placed.
    for i in mine.values() {
        for (pd, n) in i.dims.iter().zip(i.pos.shape()) {
            if point_sizes.get(pd).is_some_and(|m| m != n) {
                return Err(CoreError::DimLengthMismatch {
                    dim: pd.clone(),
                    expected: point_sizes.get(pd).copied().unwrap_or(0),
                    found: *n,
                });
            }
        }
    }
    let point_dims: Vec<Name> = point_sizes.keys().cloned().collect();
    let point_shape: Vec<usize> = point_sizes.values().copied().collect();
    let free: Vec<Name> = var
        .dims()
        .iter()
        .filter(|d| !mine.contains_key(*d))
        .cloned()
        .collect();
    let keyed: Vec<Name> = var
        .dims()
        .iter()
        .filter(|d| mine.contains_key(*d))
        .cloned()
        .collect();

    // The bounding box of the chosen positions (all a lazy variable reads).
    let mut boxes: Vec<(Name, usize, usize)> = Vec::new();
    for d in &keyed {
        let i = mine.get(d).ok_or_else(|| invalid("indexer"))?;
        let lo = i.pos.iter().copied().min().unwrap_or(0);
        let hi = i
            .pos
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m.saturating_add(1));
        boxes.push((d.clone(), lo, hi));
    }
    let to_i = |x: usize| isize::try_from(x).map_err(|_| CoreError::from(IndexError::Overflow));
    let slices = boxes
        .iter()
        .map(|(d, lo, hi)| {
            Ok((
                d.as_str(),
                Indexer::Slice(PySlice::new(Some(to_i(*lo)?), Some(to_i(*hi)?), None)),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let boxed = var.isel(slices)?;
    let typed = boxed.typed()?.to_owned();

    // Flat positions into the box's keyed axes (row-major), one per point.
    let box_sizes: Vec<usize> = boxes
        .iter()
        .map(|(_, lo, hi)| hi.saturating_sub(*lo))
        .collect();
    let mut flat = ArrayD::from_elem(IxDyn(&point_shape), 0usize);
    let mut stride = 1usize;
    for ((d, lo, _), size) in boxes.iter().zip(&box_sizes).rev() {
        let i = mine.get(d).ok_or_else(|| invalid("indexer"))?;
        let b = broadcast_to_dims(i.pos.view(), &i.dims, &point_dims)?;
        let b = b
            .broadcast(IxDyn(&point_shape))
            .ok_or_else(|| invalid("indexers do not broadcast"))?;
        for (f, p) in flat.iter_mut().zip(b.iter()) {
            let term = p
                .checked_sub(*lo)
                .and_then(|o| o.checked_mul(stride))
                .ok_or(CoreError::ArithmeticOverflow("vectorized index"))?;
            *f = f
                .checked_add(term)
                .ok_or(CoreError::ArithmeticOverflow("vectorized index"))?;
        }
        stride = stride
            .checked_mul(*size)
            .ok_or(CoreError::ArithmeticOverflow("vectorized index"))?;
    }
    let flat: Vec<usize> = flat.iter().copied().collect();

    // Keyed axes first, flattened into one; pick the points; unflatten into point dims;
    // order as `out_dims`.
    let perm: Vec<usize> = keyed
        .iter()
        .chain(&free)
        .map(|d| {
            var.dims()
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| invalid("dim"))
        })
        .collect::<Result<_>>()?;
    let free_shape: Vec<usize> = free
        .iter()
        .map(|d| boxed.len_of(d))
        .collect::<Result<_>>()?;
    let mut flat_shape = vec![stride];
    flat_shape.extend(&free_shape);
    let mut picked_shape = point_shape.clone();
    picked_shape.extend(&free_shape);
    let picked_dims: Vec<Name> = point_dims.iter().chain(&free).cloned().collect();
    let final_perm: Vec<usize> = out_dims
        .iter()
        .map(|d| {
            picked_dims
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| invalid("dim"))
        })
        .collect::<Result<_>>()?;
    let shape_err = |e: ndarray::ShapeError| invalid(e.to_string());
    let data = typed_map!(typed, a => {
        let a = a
            .permuted_axes(IxDyn(&perm))
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(IxDyn(&flat_shape))
            .map_err(shape_err)?;
        a.select(Axis(0), &flat)
            .into_shape_with_order(IxDyn(&picked_shape))
            .map_err(shape_err)?
            .permuted_axes(IxDyn(&final_perm))
            .as_standard_layout()
            .into_owned()
    });
    let mut out = Variable::from_typed(out_dims, data)?;
    out.set_unit(var.unit().cloned());
    out.attrs_mut()
        .extend(var.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

/// The indexers as positions, checked against `sizes`.
fn indexers<R: Repr>(
    sizes: &IndexMap<Name, usize>,
    given: &[(&str, &DataArrayBase<R>)],
) -> Result<IndexMap<Name, Ix>> {
    given
        .iter()
        .map(|(d, idx)| {
            let dim = Name::from(*d);
            let len = *sizes
                .get(&dim)
                .ok_or_else(|| CoreError::NoSuchDim(dim.clone()))?;
            Ok((dim.clone(), ix_from(&dim, len, idx.variable())?))
        })
        .collect()
}

/// Attach the indexers' coordinates (skipping `skip`), refusing conflicting ones.
fn attach<R: Repr>(
    coords: &mut IndexMap<Name, Variable>,
    given: &[(&str, &DataArrayBase<R>)],
    skip: &[Name],
) -> Result<()> {
    for (_, idx) in given {
        for (k, c) in idx.coords() {
            if skip.contains(k) {
                continue;
            }
            match coords.get(k) {
                Some(existing) if !variables_equal(existing, c) => {
                    return Err(invalid(format!(
                        "coordinate {k:?} of an indexer conflicts with the indexed object's"
                    )));
                }
                Some(_) => {}
                None => {
                    coords.insert(k.clone(), c.to_owned());
                }
            }
        }
    }
    Ok(())
}

/// Labels of a label array, with its unit as quantities.
fn labels_of<R: Repr>(v: &VariableBase<R>) -> Result<Vec<Label>> {
    let unit = v.unit();
    let num = |x: f64| match unit {
        Some(u) => Label::Quantity(x, u.clone()),
        None => Label::Float(x),
    };
    #[allow(clippy::cast_precision_loss)]
    Ok(match v.typed()? {
        TypedArray::F64(a) => a.iter().map(|x| num(*x)).collect(),
        TypedArray::F32(a) => a.iter().map(|x| num(f64::from(*x))).collect(),
        TypedArray::I64(a) if unit.is_none() => a.iter().map(|x| Label::Int(*x)).collect(),
        TypedArray::I32(a) if unit.is_none() => {
            a.iter().map(|x| Label::Int(i64::from(*x))).collect()
        }
        TypedArray::Str(a) => a.iter().map(|x| Label::Str(x.clone())).collect(),
        TypedArray::DateTime(a) => a.iter().map(|x| Label::DateTime(*x)).collect(),
        TypedArray::CfDate(a) => a.iter().map(|x| Label::CfDate(*x)).collect(),
        TypedArray::TimeDelta(a) => a.iter().map(|x| Label::TimeDelta(*x)).collect(),
        other => {
            let f = crate::ops::to_f64(&other, None)?;
            f.iter().map(|x| num(*x)).collect()
        }
    })
}

/// Label indexers turned into position arrays (same dimensions and coordinates).
fn positions_of<C: Repr, R: Repr>(
    coords: &IndexMap<Name, VariableBase<C>>,
    given: &[(&str, &DataArrayBase<R>)],
    method: Method,
    tolerance: Option<&Label>,
) -> Result<Vec<(Name, DataArray)>> {
    given
        .iter()
        .map(|(d, labels)| {
            let ls: Vec<LabelIndexer> = labels_of(labels.variable())?
                .into_iter()
                .map(|label| LabelIndexer::Scalar {
                    label,
                    method,
                    tolerance: tolerance.cloned(),
                })
                .collect();
            let pos = crate::select::label_positions(coords, d, ls)?;
            let pos: Vec<i64> = pos
                .into_iter()
                .map(|p| i64::try_from(p).map_err(|_| CoreError::from(IndexError::Overflow)))
                .collect::<Result<_>>()?;
            let arr = ArrayD::from_shape_vec(IxDyn(labels.variable().shape()), pos)
                .map_err(|e| invalid(e.to_string()))?;
            let var = Variable::new(labels.dims().to_vec(), arr)?;
            let coords = labels
                .coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect();
            Ok((Name::from(*d), DataArrayBase::from_parts(None, var, coords)))
        })
        .collect()
}

fn index_coords<C: Repr>(
    coords: &IndexMap<Name, VariableBase<C>>,
    ix: &IndexMap<Name, Ix>,
) -> Result<IndexMap<Name, Variable>> {
    coords
        .iter()
        .map(|(k, c)| Ok((k.clone(), vindex_var(c, ix)?)))
        .collect()
}

impl<S: Repr> DataArrayBase<S> {
    /// Pointwise selection by integer arrays with their own dimensions (xarray's `isel`
    /// with `DataArray` indexers); see the module notes.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`IndexError::OutOfBounds`]; [`CoreError::DimLengthMismatch`]
    /// for indexers that do not broadcast; [`CoreError::UnsupportedDType`] for non-integer
    /// positions; [`CoreError::InvalidReduction`] for a multi-dimensional mask or conflicting
    /// indexer coordinates.
    pub fn isel_vectorized<R: Repr>(
        &self,
        given: &[(&str, &DataArrayBase<R>)],
    ) -> Result<DataArray> {
        self.vectorized(given, &[])
    }

    fn vectorized<R: Repr>(
        &self,
        given: &[(&str, &DataArrayBase<R>)],
        skip: &[Name],
    ) -> Result<DataArray> {
        let sizes: IndexMap<Name, usize> = self.sizes().map(|(d, n)| (d.clone(), n)).collect();
        let ix = indexers(&sizes, given)?;
        let var = vindex_var(self.variable(), &ix)?;
        let mut coords = index_coords(self.coords(), &ix)?;
        attach(&mut coords, given, skip)?;
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Pointwise selection by label arrays (xarray's `sel` with `DataArray` indexers):
    /// labels become positions with `method`/`tolerance`, then as
    /// [`isel_vectorized`](Self::isel_vectorized).
    ///
    /// # Errors
    /// As for [`sel`](Self::sel) and [`isel_vectorized`](Self::isel_vectorized).
    pub fn sel_vectorized<R: Repr>(
        &self,
        given: &[(&str, &DataArrayBase<R>)],
        method: Method,
        tolerance: Option<&Label>,
    ) -> Result<DataArray> {
        let pos = positions_of(self.coords(), given, method, tolerance)?;
        let refs: Vec<(&str, &DataArray)> = pos.iter().map(|(d, a)| (d.as_str(), a)).collect();
        let skip: Vec<Name> = pos.iter().map(|(d, _)| d.clone()).collect();
        self.vectorized(&refs, &skip)
    }
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::isel_vectorized`] of every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::isel_vectorized`].
    pub fn isel_vectorized<R: Repr>(&self, given: &[(&str, &DataArrayBase<R>)]) -> Result<Dataset> {
        self.vectorized(given, &[])
    }

    fn vectorized<R: Repr>(
        &self,
        given: &[(&str, &DataArrayBase<R>)],
        skip: &[Name],
    ) -> Result<Dataset> {
        let ix = indexers(&self.sizes(), given)?;
        let mut coords = index_coords(self.coords(), &ix)?;
        attach(&mut coords, given, skip)?;
        let mut out = Dataset::new();
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), vindex_var(v, &ix)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// [`DataArrayBase::sel_vectorized`] of every variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::sel_vectorized`].
    pub fn sel_vectorized<R: Repr>(
        &self,
        given: &[(&str, &DataArrayBase<R>)],
        method: Method,
        tolerance: Option<&Label>,
    ) -> Result<Dataset> {
        let pos = positions_of(self.coords(), given, method, tolerance)?;
        let refs: Vec<(&str, &DataArray)> = pos.iter().map(|(d, a)| (d.as_str(), a)).collect();
        let skip: Vec<Name> = pos.iter().map(|(d, _)| d.clone()).collect();
        self.vectorized(&refs, &skip)
    }
}
