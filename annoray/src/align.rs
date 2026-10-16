//! Aligning data arrays and datasets on their dimension coordinates (xarray's `align`).
//!
//! For each dimension with a coordinate, the objects' labels are joined (intersection, union,
//! first, last, or required equal) and every object is reindexed onto the result. Labels
//! missing from an object become NaN (integer data turns into `f64`) or, for times, NaT;
//! strings and booleans have no missing value and give an error. Objects whose labels already
//! match are not touched (lazy data stays unread).
//!
//! Labels must have the same element type and unit to be joined: integer and float
//! coordinates, or `m` and `km`, are an error rather than an implicit cast (xarray compares
//! integers with floats by value). Convert one side explicitly.
//!
//! Ordering follows pandas: an inner join keeps the first object's order, an outer join sorts
//! (unless the indexes are equal or one is empty).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array::{Marker, marker};
use crate::dataarray::DataArrayBase;
use crate::lookup::{Keys, index_keys};
use crate::ops::to_f64;
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Element, NaT, Name, Owned, Repr, Result,
    TypedArray, Variable, VariableBase, VariableView,
};

/// How to combine the labels of a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Join {
    /// Labels present in every object (the default, as in arithmetic).
    #[default]
    Inner,
    /// Labels present in any object.
    Outer,
    /// The first object's labels.
    Left,
    /// The last object's labels.
    Right,
    /// Labels must already be equal; anything else is an error.
    Exact,
}

/// A label, comparable across objects.
#[derive(Debug, Clone)]
pub(crate) enum Key {
    F(f64),
    S(String),
    T(i128),
}

impl Key {
    /// `-0.0` and `0.0` are one label, as are all NaNs.
    fn norm_bits(v: f64) -> u64 {
        if v.is_nan() {
            f64::NAN.to_bits()
        } else if v == 0.0 {
            0
        } else {
            v.to_bits()
        }
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::F(a), Self::F(b)) => {
                if Self::norm_bits(*a) == Self::norm_bits(*b) {
                    Ordering::Equal
                } else {
                    a.total_cmp(b)
                }
            }
            (Self::S(a), Self::S(b)) => a.cmp(b),
            (Self::T(a), Self::T(b)) => a.cmp(b),
            (Self::F(_), _) | (Self::S(_), Self::T(_)) => Ordering::Less,
            _ => Ordering::Greater,
        }
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::F(v) => (0u8, Self::norm_bits(*v)).hash(state),
            Self::S(s) => (1u8, s).hash(state),
            Self::T(t) => (2u8, t).hash(state),
        }
    }
}

/// What alignment needs from one object: its dimension sizes and dimension coordinates.
pub(crate) struct Obj<'a> {
    pub(crate) sizes: IndexMap<Name, usize>,
    pub(crate) indexes: IndexMap<Name, VariableView<'a>>,
}

impl<'a> Obj<'a> {
    pub(crate) fn of_array<S: Repr>(da: &'a DataArrayBase<S>) -> Self {
        let sizes = da.sizes().map(|(d, n)| (d.clone(), n)).collect();
        Self {
            sizes,
            indexes: dim_coords(da.coords().iter()),
        }
    }

    pub(crate) fn of_dataset<S: Repr>(ds: &'a DatasetBase<S>) -> Self {
        Self {
            sizes: ds.sizes(),
            indexes: dim_coords(ds.coords().iter()),
        }
    }
}

fn dim_coords<'a, C: Repr + 'a>(
    coords: impl Iterator<Item = (&'a Name, &'a VariableBase<C>)>,
) -> IndexMap<Name, VariableView<'a>> {
    coords
        .filter(|(k, c)| c.dims().len() == 1 && c.dims().first() == Some(*k))
        .map(|(k, c)| (k.clone(), c.view()))
        .collect()
}

/// The reindexing of one dimension.
pub(crate) struct DimPlan {
    /// The joined dimension coordinate.
    coord: Variable,
    /// Per object: positions of the joined labels in it (`None` inside for a missing label),
    /// or `None` if the object is already aligned.
    positions: Vec<Option<Vec<Option<usize>>>>,
}

/// How to align a set of objects.
pub(crate) struct Plan {
    dims: IndexMap<Name, DimPlan>,
}

impl Plan {
    pub(crate) fn is_noop(&self) -> bool {
        self.dims
            .values()
            .all(|d| d.positions.iter().all(Option::is_none))
    }
}

fn cannot(dim: &Name, reason: impl Into<String>) -> CoreError {
    CoreError::CannotAlign {
        dim: dim.clone(),
        reason: reason.into(),
    }
}

/// Labels of one index, with what must match across objects for them to be comparable.
fn labels(coord: &VariableView<'_>, dim: &Name) -> Result<(Vec<Key>, String)> {
    let (keys, kind) = match index_keys(coord, dim)? {
        Keys::Num(v) => (v.into_iter().map(Key::F).collect(), "number".to_owned()),
        Keys::Str(v) => (v.into_iter().map(Key::S).collect(), "string".to_owned()),
        Keys::Ns(v, k) => (v.into_iter().map(Key::T).collect(), format!("{k:?}")),
    };
    Ok((keys, kind))
}

fn intersection(a: &[Key], b: &[Key]) -> Vec<Key> {
    let bs: HashSet<&Key> = b.iter().collect();
    let mut seen = HashSet::new();
    a.iter()
        .filter(|k| bs.contains(k) && seen.insert(*k))
        .cloned()
        .collect()
}

fn union(a: &[Key], b: &[Key]) -> Vec<Key> {
    if a == b || b.is_empty() {
        return a.to_vec();
    }
    if a.is_empty() {
        return b.to_vec();
    }
    let mut all: Vec<Key> = a.iter().chain(b).cloned().collect();
    all.sort();
    all.dedup();
    all
}

/// Copy the elements `picks` (object, position) of 1-D coordinates into one array.
fn gather(
    sources: &[(usize, TypedArray<crate::View<'_>>)],
    picks: &[(usize, usize)],
) -> Result<TypedArray<Owned>> {
    let Some((_, first)) = sources.first() else {
        return Err(CoreError::Read("alignment: no coordinates".into()));
    };
    crate::array::typed_dispatch!(first, a => gather_as(sources, picks, marker(a)))
}

fn gather_as<T: Element>(
    sources: &[(usize, TypedArray<crate::View<'_>>)],
    picks: &[(usize, usize)],
    _: Marker<T>,
) -> Result<TypedArray<Owned>> {
    let bad = || CoreError::Read("alignment: coordinate dtypes differ".into());
    let mut out = Vec::with_capacity(picks.len());
    for (obj, pos) in picks {
        let (_, src) = sources.iter().find(|(o, _)| o == obj).ok_or_else(bad)?;
        let a = T::peek(src).ok_or_else(bad)?;
        out.push(a.get(*pos).ok_or_else(bad)?.clone());
    }
    let n = out.len();
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).map_err(|_| bad())?;
    Ok(T::wrap(arr))
}

/// Work out how to align `objs`.
pub(crate) fn plan(objs: &[Obj<'_>], join: Join) -> Result<Plan> {
    let mut dims: IndexMap<Name, DimPlan> = IndexMap::new();
    let mut names: Vec<Name> = Vec::new();
    for o in objs {
        for d in o.indexes.keys() {
            if !names.contains(d) {
                names.push(d.clone());
            }
        }
    }
    for dim in names {
        // Objects with an index on this dimension.
        let holders: Vec<(usize, &VariableView<'_>)> = objs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.indexes.get(&dim).map(|c| (i, c)))
            .collect();
        let mut keyed: Vec<(usize, Vec<Key>)> = Vec::with_capacity(holders.len());
        let mut kind: Option<(String, DType, Option<annoray_units::Unit>)> = None;
        for (i, c) in &holders {
            let (k, kd) = labels(c, &dim)?;
            let this = (kd, c.dtype(), c.unit().cloned());
            match &kind {
                None => kind = Some(this),
                Some(first) if *first != this => {
                    return Err(cannot(
                        &dim,
                        format!(
                            "coordinates differ in type or unit ({} {:?} vs {} {:?}); \
                             convert one explicitly (no implicit casts between label types)",
                            first.1, first.2, this.1, this.2
                        ),
                    ));
                }
                Some(_) => {}
            }
            keyed.push((*i, k));
        }
        let all_equal = keyed.windows(2).all(|w| match w {
            [a, b] => a.1 == b.1,
            _ => true,
        });
        let joined: Vec<Key> = if all_equal {
            keyed.first().map(|(_, k)| k.clone()).unwrap_or_default()
        } else {
            match join {
                Join::Exact => return Err(CoreError::NotAligned(dim)),
                Join::Left => keyed.first().map(|(_, k)| k.clone()).unwrap_or_default(),
                Join::Right => keyed.last().map(|(_, k)| k.clone()).unwrap_or_default(),
                Join::Inner => {
                    let mut it = keyed.iter().map(|(_, k)| k.clone());
                    let first = it.next().unwrap_or_default();
                    it.fold(first, |acc, k| intersection(&acc, &k))
                }
                Join::Outer => {
                    let mut it = keyed.iter().map(|(_, k)| k.clone());
                    let first = it.next().unwrap_or_default();
                    it.fold(first, |acc, k| union(&acc, &k))
                }
            }
        };
        let n = joined.len();

        // Positions of the joined labels in each object.
        let mut positions: Vec<Option<Vec<Option<usize>>>> = vec![None; objs.len()];
        let mut lookup: Vec<(usize, HashMap<&Key, usize>)> = Vec::new();
        for (i, k) in &keyed {
            let mut map = HashMap::with_capacity(k.len());
            for (p, key) in k.iter().enumerate() {
                if map.insert(key, p).is_some() && !all_equal {
                    return Err(cannot(
                        &dim,
                        "cannot reindex a coordinate with duplicate labels",
                    ));
                }
            }
            lookup.push((*i, map));
        }
        if !all_equal {
            for ((i, k), (_, map)) in keyed.iter().zip(&lookup) {
                let pos: Vec<Option<usize>> = joined.iter().map(|j| map.get(j).copied()).collect();
                let identity =
                    pos.len() == k.len() && pos.iter().enumerate().all(|(a, b)| *b == Some(a));
                if !identity && let Some(slot) = positions.get_mut(*i) {
                    *slot = Some(pos);
                }
            }
        }
        // Objects with the dimension but no index must already have the joined length.
        for o in objs {
            if o.indexes.contains_key(&dim) {
                continue;
            }
            if let Some(&len) = o.sizes.get(&dim)
                && len != n
            {
                return Err(CoreError::DimLengthMismatch {
                    dim: dim.clone(),
                    expected: n,
                    found: len,
                });
            }
        }

        // The joined coordinate: each label taken from the first object that has it.
        let picks: Vec<(usize, usize)> = joined
            .iter()
            .map(|j| {
                lookup
                    .iter()
                    .find_map(|(i, m)| m.get(j).map(|p| (*i, *p)))
                    .ok_or_else(|| cannot(&dim, "internal: label without a source"))
            })
            .collect::<Result<_>>()?;
        let sources = holders
            .iter()
            .map(|(i, c)| c.typed().map(|t| (*i, t)))
            .collect::<Result<Vec<_>>>()?;
        let data = gather(&sources, &picks)?;
        let mut coord = Variable::from_typed([dim.clone()], data)?;
        if let Some((_, first)) = holders.first() {
            coord.set_unit(first.unit().cloned());
            coord
                .attrs_mut()
                .extend(first.attrs().iter().map(|(k, v)| (k.clone(), v.clone())));
            *coord.encoding_mut() = first.encoding().clone();
        }
        dims.insert(dim, DimPlan { coord, positions });
    }
    Ok(Plan { dims })
}

/// `v` with the elements at `pos` along `dim`, NaN where a position is `None`.
fn reindex<S: Repr>(v: &VariableBase<S>, dim: &str, pos: &[Option<usize>]) -> Result<Variable> {
    let axis = v.axis(dim)?;
    let overflow = || CoreError::ArithmeticOverflow("reindex");
    let idx = pos
        .iter()
        .map(|p| isize::try_from(p.unwrap_or(0)).map_err(|_| overflow()))
        .collect::<Result<Vec<isize>>>()?;
    let missing: Vec<usize> = pos
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.is_none().then_some(i))
        .collect();
    if missing.is_empty() {
        return v.take(dim, &idx);
    }
    let unsupported = || CoreError::UnsupportedDType {
        op: "reindexing with missing labels (needs NaN or NaT)",
        dtype: v.dtype(),
    };
    let mut shape = v.shape().to_vec();
    if let Some(s) = shape.get_mut(axis) {
        *s = pos.len();
    }
    let data = if v.len_of(dim)? == 0 {
        // Nothing to take from: all missing.
        match v.dtype() {
            DType::F32 => TypedArray::F32(ArrayD::from_elem(IxDyn(&shape), f32::NAN)),
            d if d == DType::F64 || is_int(d) => {
                TypedArray::F64(ArrayD::from_elem(IxDyn(&shape), f64::NAN))
            }
            DType::DateTime => TypedArray::DateTime(ArrayD::from_elem(IxDyn(&shape), NaT::NAT)),
            DType::TimeDelta => TypedArray::TimeDelta(ArrayD::from_elem(IxDyn(&shape), NaT::NAT)),
            DType::CfDate => TypedArray::CfDate(ArrayD::from_elem(IxDyn(&shape), NaT::NAT)),
            _ => return Err(unsupported()),
        }
    } else {
        let taken = v.take(dim, &idx)?;
        let typed = taken.typed()?;
        let mut data = match &typed {
            TypedArray::F32(a) => TypedArray::F32(a.to_owned()),
            TypedArray::F64(a) => TypedArray::F64(a.to_owned()),
            t if is_int(t.dtype()) => TypedArray::F64(to_f64(t, None)?),
            TypedArray::DateTime(a) => TypedArray::DateTime(a.to_owned()),
            TypedArray::TimeDelta(a) => TypedArray::TimeDelta(a.to_owned()),
            TypedArray::CfDate(a) => TypedArray::CfDate(a.to_owned()),
            _ => return Err(unsupported()),
        };
        for i in &missing {
            match &mut data {
                TypedArray::F32(a) => a.index_axis_mut(Axis(axis), *i).fill(f32::NAN),
                TypedArray::F64(a) => a.index_axis_mut(Axis(axis), *i).fill(f64::NAN),
                TypedArray::DateTime(a) => a.index_axis_mut(Axis(axis), *i).fill(NaT::NAT),
                TypedArray::TimeDelta(a) => a.index_axis_mut(Axis(axis), *i).fill(NaT::NAT),
                TypedArray::CfDate(a) => a.index_axis_mut(Axis(axis), *i).fill(NaT::NAT),
                _ => {}
            }
        }
        data
    };
    let mut out = Variable::from_typed(v.dims().iter().cloned(), data)?;
    out.set_unit(v.unit().cloned());
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

fn is_int(d: DType) -> bool {
    matches!(
        d,
        DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
    )
}

/// Reindex one variable (data or non-index coordinate) of object `i` along every planned dim.
fn apply_var<S: Repr>(v: &VariableBase<S>, plan: &Plan, i: usize) -> Result<Variable> {
    let mut out: Option<Variable> = None;
    for (dim, dp) in &plan.dims {
        let Some(Some(pos)) = dp.positions.get(i) else {
            continue;
        };
        if !v.dims().contains(dim) {
            continue;
        }
        out = Some(match &out {
            Some(o) => reindex(o, dim, pos)?,
            None => reindex(v, dim, pos)?,
        });
    }
    Ok(out.unwrap_or_else(|| v.to_owned()))
}

/// Coordinates of object `i`, with index coordinates replaced by the joined ones.
fn apply_coords<'c, C: Repr + 'c>(
    coords: impl Iterator<Item = (&'c Name, &'c VariableBase<C>)>,
    plan: &Plan,
    i: usize,
) -> Result<IndexMap<Name, Variable>> {
    coords
        .map(|(k, c)| {
            let joined = plan.dims.get(k).filter(|dp| {
                c.dims().len() == 1
                    && c.dims().first() == Some(k)
                    && matches!(dp.positions.get(i), Some(Some(_)))
            });
            let v = match joined {
                Some(dp) => dp.coord.clone(),
                None => apply_var(c, plan, i)?,
            };
            Ok((k.clone(), v))
        })
        .collect()
}

pub(crate) fn apply_array<S: Repr>(
    da: &DataArrayBase<S>,
    plan: &Plan,
    i: usize,
) -> Result<DataArray> {
    let var = apply_var(da.variable(), plan, i)?;
    let coords = apply_coords(da.coords().iter(), plan, i)?;
    Ok(DataArrayBase::from_parts(da.name().cloned(), var, coords))
}

pub(crate) fn apply_dataset<S: Repr>(
    ds: &DatasetBase<S>,
    plan: &Plan,
    i: usize,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    for (k, c) in apply_coords(ds.coords().iter(), plan, i)? {
        out.add_coord(k, c)?;
    }
    for (k, v) in ds.data_vars() {
        out.add_var(k.clone(), apply_var(v, plan, i)?)?;
    }
    out.attrs_mut()
        .extend(ds.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    *out.encoding_mut() = ds.encoding().clone();
    Ok(out)
}

/// Align data arrays on their dimension coordinates.
///
/// # Errors
/// [`CoreError::NotAligned`] for [`Join::Exact`] with differing labels;
/// [`CoreError::CannotAlign`] for coordinates of different types or units, duplicate labels, or
/// a dimension without a coordinate whose length disagrees; [`CoreError::UnsupportedDType`]
/// when missing labels would need NaN in non-float data.
pub fn align<S: Repr>(arrays: &[&DataArrayBase<S>], join: Join) -> Result<Vec<DataArray>> {
    let objs: Vec<Obj<'_>> = arrays.iter().map(|a| Obj::of_array(*a)).collect();
    let plan = plan(&objs, join)?;
    arrays
        .iter()
        .enumerate()
        .map(|(i, a)| apply_array(*a, &plan, i))
        .collect()
}

/// Align datasets on their dimension coordinates.
///
/// # Errors
/// As for [`align`].
pub fn align_datasets<S: Repr>(datasets: &[&DatasetBase<S>], join: Join) -> Result<Vec<Dataset>> {
    let objs: Vec<Obj<'_>> = datasets.iter().map(|d| Obj::of_dataset(*d)).collect();
    let plan = plan(&objs, join)?;
    datasets
        .iter()
        .enumerate()
        .map(|(i, d)| apply_dataset(*d, &plan, i))
        .collect()
}
