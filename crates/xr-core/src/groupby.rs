//! Split–apply–combine (xarray's `groupby` and `groupby_bins`).
//!
//! Group by a 1-D coordinate (or, for datasets, a data variable), by a date component such
//! as `"time.month"` or `"time.season"`, or by bins of a numeric coordinate. Then:
//!
//! - **reduce** each group ([`sum`](DataArrayGroupBy::sum), [`mean`](DataArrayGroupBy::mean),
//!   …): the grouped dimension is replaced by one named after the group, labelled with the
//!   group labels (sorted);
//! - take the [`first`](DataArrayGroupBy::first) or [`last`](DataArrayGroupBy::last) of each;
//! - [`map`](DataArrayGroupBy::map) a function over the groups: results that keep the
//!   grouped dimension are put back in the original order, others are stacked like a
//!   reduction;
//! - do **group arithmetic** against an array indexed by the group labels (anomalies:
//!   `(&da.groupby("time.month")? - &climatology)?`).
//!
//! Missing labels (NaN, NaT) belong to no group. Bins keep every bin. Empty groups reduce
//! like empty arrays: a sum of 0, a count of 0, NaN for a mean or a minimum (xarray gives NaN
//! for all of them). Unlike xarray, dataset variables without the grouped dimension are kept as they are
//! rather than repeated once per group.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Add, Div, Mul, Sub};

use ndarray::{ArrayD, IxDyn};
use xr_units::UnitPolicy;

use crate::align::{Join, Key, align, align_datasets};
use crate::combine::{ConcatOptions, concat, concat_datasets};
use crate::dataarray::DataArrayBase;
use crate::dt::{DtComponent, components};
use crate::lookup::{Keys, index_keys};
use crate::ops::to_f64;
use crate::{
    BinOp, CoreError, DType, DataArray, Dataset, DatasetBase, Name, ReduceOptions, Reduction, Repr,
    Result, Variable, VariableBase, VariableView,
};

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// Equal-shaped arrays of one element type stacked along a new leading axis (one copy).
fn stack_as<T: crate::Element>(
    parts: &[crate::TypedArray<crate::View<'_>>],
    _: crate::array::Marker<T>,
) -> Option<crate::TypedArray<crate::Owned>> {
    let views = parts
        .iter()
        .map(|p| T::peek(p).map(|a| a.view().insert_axis(ndarray::Axis(0))))
        .collect::<Option<Vec<_>>>()?;
    ndarray::concatenate(ndarray::Axis(0), &views)
        .ok()
        .map(T::wrap)
}

fn to_isize(v: &[usize]) -> Result<Vec<isize>> {
    v.iter()
        .map(|i| isize::try_from(*i).map_err(|_| CoreError::ArithmeticOverflow("groupby")))
        .collect()
}

/// `v` with its (single) dimension renamed.
fn renamed(v: &Variable, dim: &Name) -> Result<Variable> {
    let mut out = Variable::from_typed([dim.clone()], v.typed()?.to_owned())?;
    out.set_unit(v.unit().cloned());
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

/// How an object is split.
pub(crate) struct Grouper {
    /// The dimension grouped along.
    dim: Name,
    /// The dimension of the results (the group's name).
    name: Name,
    /// Group labels, along `name`.
    labels: Variable,
    /// Positions along `dim` of each group (empty for empty bins).
    groups: Vec<Vec<usize>>,
    /// The grouping variable along `dim` (what group arithmetic adds as a coordinate).
    key: Variable,
}

/// The grouping variable for `group`, its result name, and its missing positions.
fn grouping_var<'a>(
    lookup: impl Fn(&str) -> Option<VariableView<'a>>,
    group: &str,
) -> Result<(Name, Variable, Vec<bool>)> {
    if let Some(v) = lookup(group) {
        let n = v.len();
        return Ok((Name::from(group), v.to_owned(), vec![false; n]));
    }
    if let Some((base, comp)) = group.split_once('.')
        && let Some(v) = lookup(base)
    {
        let c: DtComponent = comp.parse()?;
        let (data, missing) = components(&v, c)?;
        let var = Variable::from_typed(v.dims().iter().cloned(), data)?;
        return Ok((Name::from(comp), var, missing));
    }
    Err(CoreError::NoSuchVariable(Name::from(group)))
}

fn one_dim(var: &Variable, group: &str) -> Result<Name> {
    match var.dims() {
        [d] => Ok(d.clone()),
        _ => Err(invalid(format!(
            "cannot group by {group:?}: it is not one-dimensional"
        ))),
    }
}

/// Labels as keys (`None` for missing ones).
fn label_keys(var: &Variable, dim: &Name, missing: &[bool]) -> Result<Vec<Option<Key>>> {
    let keys: Vec<Option<Key>> = if var.dtype() == DType::Bool {
        var.data::<bool>()?
            .iter()
            .map(|b| Some(Key::F(if *b { 1.0 } else { 0.0 })))
            .collect()
    } else {
        match index_keys(var, dim)? {
            Keys::Num(v) => v
                .into_iter()
                .map(|x| (!x.is_nan()).then_some(Key::F(x)))
                .collect(),
            Keys::Str(v) => v.into_iter().map(|s| Some(Key::S(s))).collect(),
            Keys::Ns(v, _) => v
                .into_iter()
                .map(|t| (t != i128::MIN).then_some(Key::T(t)))
                .collect(),
        }
    };
    Ok(keys
        .into_iter()
        .zip(missing)
        .map(|(k, m)| if *m { None } else { k })
        .collect())
}

impl Grouper {
    /// A grouper from precomputed groups (used by `resample`).
    pub(crate) fn new(
        dim: Name,
        name: Name,
        labels: Variable,
        groups: Vec<Vec<usize>>,
        key: Variable,
    ) -> Self {
        Self {
            dim,
            name,
            labels,
            groups,
            key,
        }
    }

    fn by_labels(name: Name, key: Variable, missing: &[bool], group: &str) -> Result<Self> {
        let dim = one_dim(&key, group)?;
        let mut map: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
        for (i, k) in label_keys(&key, &dim, missing)?.into_iter().enumerate() {
            if let Some(k) = k {
                map.entry(k).or_default().push(i);
            }
        }
        let groups: Vec<Vec<usize>> = map.into_values().collect();
        let firsts: Vec<usize> = groups.iter().filter_map(|g| g.first().copied()).collect();
        let labels = renamed(&key.take(&dim, &to_isize(&firsts)?)?, &name)?;
        Ok(Self {
            dim,
            name,
            labels,
            groups,
            key,
        })
    }

    fn by_bins(group: &str, key: Variable, bins: &[f64], opts: &BinOptions) -> Result<Self> {
        let dim = one_dim(&key, group)?;
        let n_bins = bins.len().checked_sub(1).filter(|n| *n > 0);
        let increasing = bins.windows(2).all(|w| matches!(w, [a, b] if a < b));
        let Some(n_bins) = n_bins.filter(|_| increasing) else {
            return Err(invalid("bins need at least two strictly increasing edges"));
        };
        let values = to_f64(&key.typed()?, None)?;
        let mut groups = vec![Vec::new(); n_bins];
        let first = bins.first().copied().unwrap_or(f64::NAN);
        for (i, v) in values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            // Edges below (or at) the value: the bin is the last such edge.
            let p = if opts.right {
                bins.partition_point(|e| e < v)
            } else {
                bins.partition_point(|e| e <= v)
            };
            let bin = if opts.right && opts.include_lowest && *v == first {
                Some(0)
            } else {
                p.checked_sub(1).filter(|b| *b < n_bins)
            };
            if let Some(g) = bin.and_then(|b| groups.get_mut(b)) {
                g.push(i);
            }
        }
        let edge = |e: f64| {
            if e.fract() == 0.0 && e.abs() < 1e15 {
                format!("{e:.0}")
            } else {
                format!("{e}")
            }
        };
        let (open, close) = if opts.right { ("(", "]") } else { ("[", ")") };
        let text: Vec<String> = bins
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                // With `include_lowest` the first bin is closed on both sides.
                let open = if i == 0 && opts.right && opts.include_lowest {
                    "["
                } else {
                    open
                };
                match w {
                    [a, b] => format!("{open}{}, {}{close}", edge(*a), edge(*b)),
                    _ => String::new(),
                }
            })
            .collect();
        let name = Name::from(format!("{group}_bins"));
        let labels = Variable::new(
            [name.clone()],
            ArrayD::from_shape_vec(IxDyn(&[text.len()]), text).map_err(|_| invalid("bins"))?,
        )?;
        Ok(Self {
            dim,
            name,
            labels,
            groups,
            key,
        })
    }
}

/// Options for `groupby_bins`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BinOptions {
    /// Bins include their right edge, `(a, b]` (the default), rather than their left, `[a, b)`.
    pub right: bool,
    /// With `right`, the first bin also includes its left edge.
    pub include_lowest: bool,
}

impl Default for BinOptions {
    fn default() -> Self {
        Self {
            right: true,
            include_lowest: false,
        }
    }
}

impl BinOptions {
    /// Whether bins include their right edge.
    #[must_use]
    pub const fn right(mut self, right: bool) -> Self {
        self.right = right;
        self
    }

    /// Whether the first bin includes its left edge.
    #[must_use]
    pub const fn include_lowest(mut self, include: bool) -> Self {
        self.include_lowest = include;
        self
    }
}

// ----- the combine step, shared by data arrays and datasets -----

/// What the combine step needs from results (data arrays or datasets).
pub(crate) trait GroupOut: Sized {
    fn reduce_along(&self, op: Reduction, dim: &Name, opts: &ReduceOptions) -> Result<Self>;
    fn index0(&self, dim: &Name) -> Result<Self>;
    fn has_dim(&self, dim: &Name) -> bool;
    fn len_along(&self, dim: &Name) -> usize;
    fn concat_along(parts: &[Self], dim: &Name) -> Result<Self>;
    /// Stack results of the same shape along a new leading dimension `dim`. By default a
    /// concatenation; data arrays copy the values once, without concatenation's checks.
    fn stack_new(parts: &[Self], dim: &Name) -> Result<Self> {
        Self::concat_along(parts, dim)
    }
    fn take_along(&self, dim: &Name, idx: &[usize]) -> Result<Self>;
    fn set_coord(self, name: &Name, v: Variable) -> Result<Self>;
    fn has_var(&self, name: &Name) -> bool;
    /// Reindex onto all `labels` (along `name`), NaN where a label has no result.
    fn reindexed(self, labels: &Variable, name: &Name) -> Result<Self>;
}

impl GroupOut for DataArray {
    fn reduce_along(&self, op: Reduction, dim: &Name, opts: &ReduceOptions) -> Result<Self> {
        self.reduce(op, vec![dim.clone()], opts)
    }
    fn index0(&self, dim: &Name) -> Result<Self> {
        Ok(self.isel([(dim.as_str(), 0)])?.to_owned())
    }
    fn has_dim(&self, dim: &Name) -> bool {
        self.dims().contains(dim)
    }
    fn len_along(&self, dim: &Name) -> usize {
        self.len_of(dim).unwrap_or(0)
    }
    fn concat_along(parts: &[Self], dim: &Name) -> Result<Self> {
        let refs: Vec<&Self> = parts.iter().collect();
        concat(&refs, dim, &ConcatOptions::default())
    }
    fn stack_new(parts: &[Self], dim: &Name) -> Result<Self> {
        let Some(first) = parts.first() else {
            return Err(invalid("no groups"));
        };
        let same = parts.iter().all(|p| {
            p.dims() == first.dims()
                && p.shape() == first.shape()
                && p.dtype() == first.dtype()
                && p.unit() == first.unit()
                && !p.dims().contains(dim)
                && p.coords().len() == first.coords().len()
                && p.coords().iter().all(|(k, c)| {
                    first
                        .coords()
                        .get(k)
                        .is_some_and(|f| crate::variables_equal(c, f))
                })
        });
        if !same {
            return Self::concat_along(parts, dim);
        }
        let views = parts
            .iter()
            .map(|p| p.variable().typed())
            .collect::<Result<Vec<_>>>()?;
        let Some(head) = views.first() else {
            return Err(invalid("no groups"));
        };
        let Some(data) =
            crate::array::typed_dispatch!(head, x => stack_as(&views, crate::array::marker(x)))
        else {
            return Self::concat_along(parts, dim);
        };
        let dims: Vec<Name> = std::iter::once(dim.clone())
            .chain(first.dims().iter().cloned())
            .collect();
        let mut var = Variable::from_typed(dims, data)?;
        var.set_unit(first.unit().cloned());
        var.attrs_mut()
            .extend(first.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        let coords = first
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            first.name().cloned(),
            var,
            coords,
        ))
    }
    fn take_along(&self, dim: &Name, idx: &[usize]) -> Result<Self> {
        self.take(dim, &to_isize(idx)?)
    }
    fn set_coord(self, name: &Name, v: Variable) -> Result<Self> {
        let (n, var, mut coords) = self.into_parts();
        coords.insert(name.clone(), v);
        Ok(DataArrayBase::from_parts(n, var, coords))
    }
    fn has_var(&self, name: &Name) -> bool {
        self.coords().contains_key(name)
    }
    fn reindexed(self, labels: &Variable, name: &Name) -> Result<Self> {
        let n = labels.len();
        let dummy = DataArray::new(Variable::new(
            [name.clone()],
            ArrayD::from_elem(IxDyn(&[n]), 0.0),
        )?)
        .set_coord(name, labels.clone())?;
        align(&[&self, &dummy], Join::Right)?
            .into_iter()
            .next()
            .ok_or_else(|| invalid("reindex"))
    }
}

impl GroupOut for Dataset {
    fn reduce_along(&self, op: Reduction, dim: &Name, opts: &ReduceOptions) -> Result<Self> {
        self.reduce(op, vec![dim.clone()], opts)
    }
    fn index0(&self, dim: &Name) -> Result<Self> {
        Ok(self.isel([(dim.as_str(), 0)])?.to_owned())
    }
    fn has_dim(&self, dim: &Name) -> bool {
        self.sizes().contains_key(dim)
    }
    fn len_along(&self, dim: &Name) -> usize {
        self.sizes().get(dim).copied().unwrap_or(0)
    }
    fn concat_along(parts: &[Self], dim: &Name) -> Result<Self> {
        let refs: Vec<&Self> = parts.iter().collect();
        concat_datasets(&refs, dim, &ConcatOptions::default())
    }
    fn take_along(&self, dim: &Name, idx: &[usize]) -> Result<Self> {
        self.take(dim, &to_isize(idx)?)
    }
    fn set_coord(mut self, name: &Name, v: Variable) -> Result<Self> {
        if self.contains(name) {
            self.remove(name)?;
        }
        self.add_coord(name.clone(), v)?;
        Ok(self)
    }
    fn has_var(&self, name: &Name) -> bool {
        self.contains(name)
    }
    fn reindexed(self, labels: &Variable, name: &Name) -> Result<Self> {
        let dummy = Dataset::new().with_coord(name.clone(), labels.clone())?;
        align_datasets(&[&self, &dummy], Join::Right)?
            .into_iter()
            .next()
            .ok_or_else(|| invalid("reindex"))
    }
}

/// Stack per-group results (for the groups `kept`) along the group dimension.
fn stacked<O: GroupOut>(g: &Grouper, parts: Vec<O>, kept: &[usize]) -> Result<O> {
    if parts.is_empty() {
        return Err(invalid("no groups"));
    }
    let out = O::stack_new(&parts, &g.name)?;
    let labels = g.labels.take(&g.name, &to_isize(kept)?)?;
    let out = out.set_coord(&g.name, labels)?;
    if kept.len() == g.groups.len() {
        Ok(out)
    } else {
        out.reindexed(&g.labels, &g.name)
    }
}

/// Concatenate per-group results that keep the grouped dimension, back in original order.
fn restored<O: GroupOut>(g: &Grouper, parts: Vec<O>, kept: &[usize]) -> Result<O> {
    let out = O::concat_along(&parts, &g.dim)?;
    let order: Vec<usize> = kept
        .iter()
        .filter_map(|i| g.groups.get(*i))
        .flatten()
        .copied()
        .collect();
    if out.len_along(&g.dim) != order.len() {
        // The function changed the groups' lengths: keep the concatenation order.
        return Ok(out);
    }
    let mut perm: Vec<usize> = (0..order.len()).collect();
    perm.sort_by_key(|p| order.get(*p).copied().unwrap_or(usize::MAX));
    let out = out.take_along(&g.dim, &perm)?;
    // The grouping variable becomes a coordinate (e.g. `season` along `time`).
    if out.has_var(&g.name) || g.key.dims().first() != Some(&g.dim) {
        return Ok(out);
    }
    let mut sorted = order;
    sorted.sort_unstable();
    let key = g.key.take(&g.dim, &to_isize(&sorted)?)?;
    out.set_coord(&g.name, key)
}

fn nonempty(g: &Grouper) -> impl Iterator<Item = (usize, &Vec<usize>)> {
    g.groups
        .iter()
        .enumerate()
        .filter(|(_, idx)| !idx.is_empty())
}

/// The range `start..end` if `idx` is one (ascending, without gaps).
fn contiguous(idx: &[usize]) -> Option<(usize, usize)> {
    let first = *idx.first()?;
    let ok = idx
        .iter()
        .enumerate()
        .all(|(k, i)| first.checked_add(k) == Some(*i));
    ok.then(|| (first, first.saturating_add(idx.len())))
}

/// `dim = start..end` as a positional slice.
fn range_indexer(dim: &Name, (start, end): (usize, usize)) -> Result<[(Name, crate::Indexer); 1]> {
    let s = isize::try_from(start).map_err(|_| CoreError::ArithmeticOverflow("groupby"))?;
    let e = isize::try_from(end).map_err(|_| CoreError::ArithmeticOverflow("groupby"))?;
    Ok([(
        dim.clone(),
        crate::Indexer::Slice(crate::PySlice::new(Some(s), Some(e), None)),
    )])
}

/// Reduce every group (`one` reduces the group at the given positions).
fn reduce_groups<O: GroupOut>(g: &Grouper, one: impl Fn(&[usize]) -> Result<O>) -> Result<O> {
    let mut parts = Vec::new();
    let mut kept = Vec::new();
    for (i, idx) in g.groups.iter().enumerate() {
        // An empty group reduces like any empty array (a sum of 0, a count of 0, a NaN
        // mean); where that is undefined (the minimum of nothing) it becomes NaN.
        match one(idx) {
            Ok(r) => {
                parts.push(r);
                kept.push(i);
            }
            Err(CoreError::EmptyReduction(_)) if idx.is_empty() => {}
            Err(e) => return Err(e),
        }
    }
    stacked(g, parts, &kept)
}

fn pick_groups<O: GroupOut>(
    g: &Grouper,
    get: impl Fn(&[usize]) -> Result<O>,
    last: bool,
) -> Result<O> {
    let mut parts = Vec::new();
    let mut kept = Vec::new();
    for (i, idx) in nonempty(g) {
        let pos = if last { idx.last() } else { idx.first() };
        let Some(pos) = pos else { continue };
        parts.push(get(&[*pos])?.index0(&g.dim)?);
        kept.push(i);
    }
    stacked(g, parts, &kept)
}

fn map_groups<O: GroupOut>(
    g: &Grouper,
    get: impl Fn(&[usize]) -> Result<O>,
    mut f: impl FnMut(usize, O) -> Result<O>,
) -> Result<O> {
    let mut parts = Vec::new();
    let mut kept = Vec::new();
    for (i, idx) in nonempty(g) {
        parts.push(f(i, get(idx)?)?);
        kept.push(i);
    }
    if !parts.is_empty() && parts.iter().all(|p| p.has_dim(&g.dim)) {
        restored(g, parts, &kept)
    } else {
        stacked(g, parts, &kept)
    }
}

// ----- data arrays -----

/// A data array split into groups. Made by [`DataArrayBase::groupby`] and
/// [`DataArrayBase::groupby_bins`].
pub struct DataArrayGroupBy<'a, S: Repr> {
    obj: &'a DataArrayBase<S>,
    g: Grouper,
}

/// A dataset split into groups. Made by [`DatasetBase::groupby`] and
/// [`DatasetBase::groupby_bins`].
pub struct DatasetGroupBy<'a, S: Repr> {
    obj: &'a DatasetBase<S>,
    g: Grouper,
}

macro_rules! common {
    ($out:ty) => {
        /// Number of groups.
        #[must_use]
        pub fn len(&self) -> usize {
            self.g.groups.len()
        }

        /// Whether there are no groups.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.g.groups.is_empty()
        }

        /// The group labels, sorted (for bins: every bin, in order), along the group dimension.
        #[must_use]
        pub fn labels(&self) -> &Variable {
            &self.g.labels
        }

        /// Positions along the grouped dimension of each group.
        #[must_use]
        pub fn indices(&self) -> &[Vec<usize>] {
            &self.g.groups
        }

        /// The dimension being grouped along.
        #[must_use]
        pub fn dim(&self) -> &Name {
            &self.g.dim
        }

        /// The name of the group dimension in results.
        #[must_use]
        pub fn name(&self) -> &Name {
            &self.g.name
        }

        /// Group `i` (in label order).
        ///
        /// # Errors
        /// [`CoreError::InvalidReduction`] if there is no such group; read errors for lazy
        /// data.
        pub fn group(&self, i: usize) -> Result<$out> {
            let idx = self
                .g
                .groups
                .get(i)
                .ok_or_else(|| invalid(format!("no group {i}")))?;
            self.get(idx)
        }

        /// Reduce each group over the grouped dimension.
        ///
        /// # Errors
        /// As for the reduction itself; [`CoreError::InvalidReduction`] if every group is
        /// empty.
        pub fn reduce(&self, op: Reduction, opts: &ReduceOptions) -> Result<$out> {
            self.reduce_impl(op, opts)
        }

        /// The first element of each group.
        ///
        /// # Errors
        /// [`CoreError::InvalidReduction`] if every group is empty.
        pub fn first(&self) -> Result<$out> {
            self.finish(pick_groups(&self.g, |idx| self.get(idx), false)?)
        }

        /// The last element of each group.
        ///
        /// # Errors
        /// [`CoreError::InvalidReduction`] if every group is empty.
        pub fn last(&self) -> Result<$out> {
            self.finish(pick_groups(&self.g, |idx| self.get(idx), true)?)
        }

        /// Apply `f` to each group and combine: results that keep the grouped dimension are
        /// put back in the original order; others are stacked along the group dimension.
        ///
        /// # Errors
        /// Whatever `f` returns, or combining errors.
        pub fn map(&self, mut f: impl FnMut($out) -> Result<$out>) -> Result<$out> {
            map_groups(&self.g, |idx| self.get(idx), |_, x| f(x))
        }

        group_shortcuts!($out);
    };
}

macro_rules! group_shortcuts {
    ($out:ty) => {
        group_shortcuts!(@ $out;
            sum Sum "Sum of each group.",
            prod Prod "Product of each group.",
            mean Mean "Mean of each group.",
            median Median "Median of each group.",
            std Std "Standard deviation of each group (`ddof = 0`).",
            var Var "Variance of each group (`ddof = 0`).",
            min Min "Minimum of each group.",
            max Max "Maximum of each group.",
            count Count "Number of non-missing values in each group.",
            any Any "Whether any value of each group is non-zero.",
            all All "Whether every value of each group is non-zero.",
        );
    };
    (@ $out:ty; $($m:ident $op:ident $doc:literal),* $(,)?) => {$(
        #[doc = $doc]
        ///
        /// NaNs are skipped. See [`reduce`](Self::reduce) for options.
        ///
        /// # Errors
        /// As for [`reduce`](Self::reduce).
        pub fn $m(&self) -> Result<$out> {
            self.reduce(Reduction::$op, &ReduceOptions::default())
        }
    )*};
}

impl<'a, S: Repr> DataArrayGroupBy<'a, S> {
    pub(crate) fn from_grouper(obj: &'a DataArrayBase<S>, g: Grouper) -> Self {
        Self { obj, g }
    }

    fn get(&self, idx: &[usize]) -> Result<DataArray> {
        self.obj.take(&self.g.dim, &to_isize(idx)?)
    }

    fn reduce_impl(&self, op: Reduction, opts: &ReduceOptions) -> Result<DataArray> {
        let dim = &self.g.dim;
        let out = reduce_groups(&self.g, |idx| match contiguous(idx) {
            // Runs (all of resample's groups): reduce a view, copying nothing.
            Some(r) => self
                .obj
                .isel(range_indexer(dim, r)?)?
                .reduce(op, vec![dim.clone()], opts),
            None => self.get(idx)?.reduce_along(op, dim, opts),
        })?;
        self.finish(out)
    }

    /// The group dimension where the grouped one was, as xarray does for data arrays
    /// (datasets put it first, also as xarray does).
    fn finish(&self, out: DataArray) -> Result<DataArray> {
        let mut order: Vec<&str> = self
            .obj
            .dims()
            .iter()
            .map(|d| {
                if *d == self.g.dim {
                    self.g.name.as_str()
                } else {
                    d.as_str()
                }
            })
            .filter(|d| out.dims().iter().any(|x| x == d))
            .collect();
        for d in out.dims() {
            if !order.contains(&d.as_str()) {
                order.push(d.as_str());
            }
        }
        if out
            .dims()
            .iter()
            .map(Name::as_str)
            .eq(order.iter().copied())
        {
            return Ok(out);
        }
        out.transpose(&order)
    }

    common!(DataArray);

    /// `group op other` for every group, where `other` is indexed by the group labels along
    /// the group dimension (e.g. a climatology by month). The result has this array's shape
    /// and order, with the grouping variable as a coordinate.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] if `other` lacks the group dimension;
    /// [`CoreError::LabelNotFound`] if a group label is missing from it; arithmetic errors.
    pub fn binary<R: Repr>(
        &self,
        op: BinOp,
        other: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<DataArray> {
        // `other` expanded along the grouped dimension (each position gets its group's entry)
        // with one take, then one elementwise operation: no per-group copies or reassembly.
        let name = &self.g.name;
        let dim = &self.g.dim;
        let coord = other.coord(name)?.to_owned();
        let theirs: HashMap<Key, usize> = label_keys(&coord, name, &vec![false; coord.len()])?
            .into_iter()
            .enumerate()
            .filter_map(|(i, k)| k.map(|k| (k, i)))
            .collect();
        let ours = label_keys(&self.g.labels, name, &vec![false; self.g.labels.len()])?;
        let n = self.obj.len_of(dim)?;
        let mut pos: Vec<Option<usize>> = vec![None; n];
        for (gi, idx) in self.g.groups.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let j = ours
                .get(gi)
                .and_then(Option::as_ref)
                .and_then(|k| theirs.get(k))
                .ok_or_else(|| CoreError::LabelNotFound {
                    dim: name.clone(),
                    label: format!("group {gi}"),
                })?;
            for i in idx {
                if let Some(p) = pos.get_mut(*i) {
                    *p = Some(*j);
                }
            }
        }
        let Some(pos) = pos.into_iter().collect::<Option<Vec<usize>>>() else {
            // Positions outside every group (missing labels): group by group.
            return self.binary_by_groups(op, other, policy);
        };
        let var = other
            .variable()
            .take(name, &to_isize(&pos)?)?
            .rename_dim(name, dim)?;
        let mut rhs = DataArray::new(var);
        for (k, c) in other.coords() {
            if !c.dims().contains(name) {
                rhs = rhs.with_coord(k.clone(), c.to_owned())?;
            }
        }
        let mut out = self.obj.binary(op, &rhs, policy)?;
        // The grouping variable as a coordinate, as xarray adds it.
        if !out.coords().contains_key(name) {
            out = out.with_coord(name.clone(), self.g.key.clone())?;
        }
        Ok(out)
    }

    fn binary_by_groups<R: Repr>(
        &self,
        op: BinOp,
        other: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<DataArray> {
        let name = &self.g.name;
        let coord = other.coord(name)?.to_owned();
        let theirs: HashMap<Key, usize> = label_keys(&coord, name, &vec![false; coord.len()])?
            .into_iter()
            .enumerate()
            .filter_map(|(i, k)| k.map(|k| (k, i)))
            .collect();
        let ours = label_keys(&self.g.labels, name, &vec![false; self.g.labels.len()])?;
        map_groups(
            &self.g,
            |idx| self.get(idx),
            |i, part| {
                let j = ours
                    .get(i)
                    .and_then(Option::as_ref)
                    .and_then(|k| theirs.get(k))
                    .ok_or_else(|| CoreError::LabelNotFound {
                        dim: name.clone(),
                        label: format!("group {i}"),
                    })?;
                let (n, var, mut coords) =
                    other.isel([(name.as_str(), *j)])?.to_owned().into_parts();
                coords.shift_remove(name);
                let rhs = DataArrayBase::from_parts(n, var, coords);
                part.binary(op, &rhs, policy)
            },
        )
    }
}

macro_rules! group_ops {
    ($($tr:ident $m:ident $op:ident),*) => {$(
        impl<S: Repr, R: Repr> $tr<&DataArrayBase<R>> for &DataArrayGroupBy<'_, S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
                self.binary(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
    )*};
}
group_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div);

impl<'a, S: Repr> DatasetGroupBy<'a, S> {
    pub(crate) fn from_grouper(obj: &'a DatasetBase<S>, g: Grouper) -> Self {
        Self { obj, g }
    }

    fn get(&self, idx: &[usize]) -> Result<Dataset> {
        self.obj.take(&self.g.dim, &to_isize(idx)?)
    }

    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn finish(&self, out: Dataset) -> Result<Dataset> {
        Ok(out)
    }

    fn reduce_impl(&self, op: Reduction, opts: &ReduceOptions) -> Result<Dataset> {
        // Only variables with the grouped dimension are grouped; the rest are kept as they
        // are (numeric reductions drop non-numeric ones, as `Dataset::reduce` does).
        let dim = &self.g.dim;
        let mut sub = self.obj.to_owned();
        let mut kept: Vec<(Name, Variable)> = Vec::new();
        for (k, v) in self.obj.data_vars() {
            if !v.dims().contains(dim) {
                sub.remove(k)?;
                let numeric = !matches!(
                    v.dtype(),
                    DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta
                );
                let numeric_only = matches!(
                    op,
                    Reduction::Sum
                        | Reduction::Prod
                        | Reduction::Mean
                        | Reduction::Median
                        | Reduction::Std
                        | Reduction::Var
                );
                if numeric || !numeric_only {
                    kept.push((k.clone(), v.to_owned()));
                }
            }
        }
        let mut out = reduce_groups(&self.g, |idx| match contiguous(idx) {
            Some(r) => sub
                .isel(range_indexer(dim, r)?)?
                .reduce(op, vec![dim.clone()], opts),
            None => sub.take(dim, &to_isize(idx)?)?.reduce_along(op, dim, opts),
        })?;
        for (k, v) in kept {
            out.add_var(k, v)?;
        }
        Ok(out)
    }

    common!(Dataset);
}

// ----- constructors -----

impl<S: Repr> DataArrayBase<S> {
    /// Split by the values of a 1-D coordinate, or by a date component of one
    /// (`"time.month"`, `"time.season"`, …; see [`VariableBase::dt`]).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] for an unknown coordinate; [`CoreError::InvalidReduction`]
    /// for a coordinate that is not one-dimensional or an unknown component.
    pub fn groupby(&self, group: &str) -> Result<DataArrayGroupBy<'_, S>> {
        let (name, key, missing) =
            grouping_var(|n| self.coords().get(n).map(VariableBase::view), group)?;
        Ok(DataArrayGroupBy {
            obj: self,
            g: Grouper::by_labels(name, key, &missing, group)?,
        })
    }

    /// Split by bins of a numeric 1-D coordinate, with `bins` edges. The group dimension is
    /// `"<group>_bins"`, labelled `"(a, b]"`.
    ///
    /// # Errors
    /// As for [`groupby`](Self::groupby); [`CoreError::InvalidReduction`] for fewer than two
    /// edges or edges that do not increase; [`CoreError::UnsupportedDType`] for non-numeric
    /// coordinates.
    pub fn groupby_bins(
        &self,
        group: &str,
        bins: &[f64],
        opts: &BinOptions,
    ) -> Result<DataArrayGroupBy<'_, S>> {
        let (_, key, _) = grouping_var(|n| self.coords().get(n).map(VariableBase::view), group)?;
        Ok(DataArrayGroupBy {
            obj: self,
            g: Grouper::by_bins(group, key, bins, opts)?,
        })
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Split by the values of a 1-D coordinate or data variable, or by a date component of
    /// one (`"time.month"`, …).
    ///
    /// # Errors
    /// As for [`DataArrayBase::groupby`].
    pub fn groupby(&self, group: &str) -> Result<DatasetGroupBy<'_, S>> {
        let (name, key, missing) = grouping_var(
            |n| {
                self.coords()
                    .get(n)
                    .map(VariableBase::view)
                    .or_else(|| self.data_vars().get(n).map(VariableBase::view))
            },
            group,
        )?;
        Ok(DatasetGroupBy {
            obj: self,
            g: Grouper::by_labels(name, key, &missing, group)?,
        })
    }

    /// Split by bins of a numeric 1-D coordinate or data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::groupby_bins`].
    pub fn groupby_bins(
        &self,
        group: &str,
        bins: &[f64],
        opts: &BinOptions,
    ) -> Result<DatasetGroupBy<'_, S>> {
        let (_, key, _) = grouping_var(
            |n| {
                self.coords()
                    .get(n)
                    .map(VariableBase::view)
                    .or_else(|| self.data_vars().get(n).map(VariableBase::view))
            },
            group,
        )?;
        Ok(DatasetGroupBy {
            obj: self,
            g: Grouper::by_bins(group, key, bins, opts)?,
        })
    }
}
