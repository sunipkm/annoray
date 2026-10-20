//! Block reductions (xarray's `coarsen`): each coarsened dimension is cut into consecutive
//! windows that are reduced to one value.
//!
//! Every coarsened dimension is split into (blocks, window) and the window dimensions are
//! reduced together with [`reduce`](crate::DataArrayBase::reduce), so every reduction, NaN
//! handling and unit rule of `reduce` applies. Coordinates along a coarsened dimension are
//! reduced with the options' coordinate reduction (the mean, as xarray's `coord_func`);
//! non-numeric ones take each window's first label. Lazy data is read when reducing, slab by
//! slab within the streaming budget.

use indexmap::IndexMap;
use ndarray::IxDyn;

use crate::{
    CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Indexer, Name, Owned,
    PySlice, ReduceOptions, Reduction, Repr, Result, TypedArray, Variable, VariableBase,
};

/// What to do when a dimension's length is not a multiple of its window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Boundary {
    /// An error (the default).
    #[default]
    Exact,
    /// Drop the incomplete window.
    Trim,
    /// Complete the last window with NaN (integers become `f64`).
    Pad,
}

/// Where windows start (and so which end a trim or pad affects).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Side {
    /// Windows start at the first element; the remainder is at the end (the default).
    #[default]
    Left,
    /// Windows end at the last element; the remainder is at the start.
    Right,
}

/// Options for [`coarsen`](DataArrayBase::coarsen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CoarsenOptions {
    /// Incomplete windows.
    pub boundary: Boundary,
    /// Window alignment.
    pub side: Side,
    /// How numeric coordinates along coarsened dimensions are reduced.
    pub coord_reduction: Reduction,
}

impl Default for CoarsenOptions {
    fn default() -> Self {
        Self {
            boundary: Boundary::Exact,
            side: Side::Left,
            coord_reduction: Reduction::Mean,
        }
    }
}

impl CoarsenOptions {
    /// Incomplete windows.
    #[must_use]
    pub const fn boundary(mut self, boundary: Boundary) -> Self {
        self.boundary = boundary;
        self
    }

    /// Window alignment.
    #[must_use]
    pub const fn side(mut self, side: Side) -> Self {
        self.side = side;
        self
    }

    /// How numeric coordinates along coarsened dimensions are reduced.
    #[must_use]
    pub const fn coord_reduction(mut self, op: Reduction) -> Self {
        self.coord_reduction = op;
        self
    }
}

/// A data array to reduce in windows: from [`DataArrayBase::coarsen`]. Lazy data stays lazy
/// until a reduction, which reads it slab by slab within the streaming budget.
#[derive(Debug, Clone)]
pub struct Coarsen {
    /// Trimmed or padded to whole windows (lazily, for lazy data).
    source: DataArray,
    sizes: Vec<(Name, usize)>,
    opts: CoarsenOptions,
    order: Vec<Name>,
}

/// Data and coordinates cut into windows.
#[derive(Debug, Clone)]
struct Cut {
    data: DataArray,
    /// Each coarsened dimension and its window dimension.
    windows: Vec<(Name, Name)>,
    window_dims: Vec<Name>,
    /// Coordinates along coarsened dimensions, reduced to one value per window.
    coords: IndexMap<Name, Variable>,
    /// The same coordinates cut into windows (for `construct`).
    split_coords: IndexMap<Name, Variable>,
    order: Vec<Name>,
}

fn window_name(dim: &Name) -> Name {
    Name::from(format!("{dim}__window"))
}

fn invalid(msg: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(msg.into())
}

/// `v` with axis `dim` of length `k·w` split into `dim` (`k`) and `window` (`w`).
fn split<S: Repr>(v: &VariableBase<S>, dim: &Name, w: usize, window: &Name) -> Result<Variable> {
    let axis = v.axis(dim)?;
    let mut shape = v.shape().to_vec();
    let len = shape.get(axis).copied().unwrap_or(0);
    let k = len.checked_div(w).ok_or_else(|| invalid("a window of 0"))?;
    if let Some(s) = shape.get_mut(axis) {
        *s = k;
    }
    shape.insert(axis.saturating_add(1), w);
    let mut dims = v.dims().to_vec();
    dims.insert(axis.saturating_add(1), window.clone());
    let typed: TypedArray<Owned> = v.typed()?.to_owned();
    let data = crate::array::typed_dispatch!(typed, a => crate::array::Element::wrap::<Owned>(
        a.as_standard_layout()
            .into_owned()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| invalid(e.to_string()))?
    ));
    let mut out = Variable::from_typed(dims, data)?;
    out.set_unit(v.unit().cloned());
    out.take_zone_from(v);
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

/// `da` trimmed or padded along `dim` to a multiple of `w`.
fn fit<S: Repr>(
    da: &DataArrayBase<S>,
    dim: &Name,
    w: usize,
    opts: &CoarsenOptions,
) -> Result<DataArray> {
    let n = da.variable().len_of(dim)?;
    let rem = n.checked_rem(w).ok_or_else(|| invalid("a window of 0"))?;
    if rem == 0 {
        return Ok(da.to_owned());
    }
    let to_i = |x: usize| isize::try_from(x).map_err(|_| CoreError::ArithmeticOverflow("coarsen"));
    match opts.boundary {
        Boundary::Exact => Err(invalid(format!(
            "coarsen: {dim:?} has length {n}, not a multiple of the window {w} (use trim or pad)"
        ))),
        Boundary::Trim => {
            let (start, stop) = match opts.side {
                Side::Left => (None, Some(to_i(n.saturating_sub(rem))?)),
                Side::Right => (Some(to_i(rem)?), None),
            };
            Ok(da
                .isel([(
                    dim.as_str(),
                    Indexer::Slice(PySlice::new(start, stop, None)),
                )])?
                .to_owned())
        }
        Boundary::Pad => {
            let extra = w.saturating_sub(rem);
            let total = n.saturating_add(extra);
            let (lead, n_i) = match opts.side {
                Side::Left => (0, to_i(n)?),
                Side::Right => (to_i(extra)?, to_i(n)?),
            };
            let last = n_i.saturating_sub(1).max(0);
            let idx: Vec<isize> = (0..to_i(total)?)
                .map(|i| i.saturating_sub(lead).clamp(0, last))
                .collect();
            let valid: Vec<bool> = (0..to_i(total)?)
                .map(|i| {
                    let j = i.saturating_sub(lead);
                    j >= 0 && j < n_i
                })
                .collect();
            let cond = DataArray::from_array(
                [dim.as_str()],
                ndarray::ArrayD::from_shape_vec(IxDyn(&[total]), valid)
                    .map_err(|e| invalid(e.to_string()))?,
            )?;
            // Pad data and coordinates alike (coordinates become NaN there too, and are
            // skipped by the coordinate reduction).
            let taken = da.take(dim, &idx)?;
            let data = taken.where_(&cond)?;
            let mut coords = IndexMap::new();
            for (k, c) in taken.coords() {
                let c = if c.dims().contains(dim) && numeric(c.dtype()) {
                    DataArrayBase::from_parts(None, c.clone(), IndexMap::new())
                        .where_(&cond)?
                        .variable()
                        .to_owned()
                } else {
                    c.clone()
                };
                coords.insert(k.clone(), c);
            }
            Ok(DataArrayBase::from_parts(
                data.name().cloned(),
                data.variable().to_owned(),
                coords,
            ))
        }
    }
}

fn numeric(d: DType) -> bool {
    !matches!(d, DType::Str | DType::Bool | DType::CfDate)
}

impl<S: Repr> DataArrayBase<S> {
    /// Cut `windows` (dimension, window length) into blocks for reduction:
    /// `da.coarsen(&[("time", 4)], &CoarsenOptions::default())?.mean()`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] for a window of 0, a
    /// repeated dimension, or (with [`Boundary::Exact`]) a length that is not a multiple of
    /// the window.
    pub fn coarsen(&self, windows: &[(&str, usize)], opts: &CoarsenOptions) -> Result<Coarsen> {
        let order: Vec<Name> = self.coords().keys().cloned().collect();
        // Fit every dimension first (data and coordinates together), then cut each into
        // windows, so coordinates along several coarsened dimensions are cut along all.
        let mut da = self.to_owned();
        let mut pairs: Vec<(Name, Name)> = Vec::new();
        for (dim, w) in windows {
            let dim = Name::from(*dim);
            if *w == 0 {
                return Err(invalid("a window of 0"));
            }
            if pairs.iter().any(|(d, _)| *d == dim) {
                return Err(invalid(format!("coarsen: {dim:?} given twice")));
            }
            da = fit(&da, &dim, *w, opts)?;
            pairs.push((dim.clone(), window_name(&dim)));
        }
        Ok(Coarsen {
            source: da,
            sizes: pairs
                .iter()
                .zip(windows)
                .map(|((d, _), (_, w))| (d.clone(), *w))
                .collect(),
            opts: *opts,
            order,
        })
    }
}

/// `da` (already whole windows long) cut along `sizes`.
fn cut(
    da: &DataArray,
    sizes: &[(Name, usize)],
    opts: &CoarsenOptions,
    order: &[Name],
) -> Result<Cut> {
    let pairs: Vec<(Name, Name)> = sizes
        .iter()
        .map(|(d, _)| (d.clone(), window_name(d)))
        .collect();
    let order = order.to_vec();
    let mut var = da.variable().to_owned();
    let mut kept = IndexMap::new();
    let mut split_coords: IndexMap<Name, Variable> = IndexMap::new();
    for (k, c) in da.coords() {
        if pairs.iter().any(|(d, _)| c.dims().contains(d)) {
            split_coords.insert(k.clone(), c.clone());
        } else {
            kept.insert(k.clone(), c.clone());
        }
    }
    for ((dim, window), (_, w)) in pairs.iter().zip(sizes) {
        var = split(&var, dim, *w, window)?;
        for c in split_coords.values_mut() {
            if c.dims().contains(dim) {
                *c = split(c, dim, *w, window)?;
            }
        }
    }
    let mut coords = IndexMap::new();
    for (k, c) in &split_coords {
        let own: Vec<Name> = pairs
            .iter()
            .map(|(_, wd)| wd.clone())
            .filter(|wd| c.dims().contains(wd))
            .collect();
        let reduced = if numeric(c.dtype()) {
            c.reduce(opts.coord_reduction, own, &ReduceOptions::default())?
        } else {
            c.isel(own.iter().map(|wd| (wd.as_str(), 0isize)))?
                .to_owned()
        };
        coords.insert(k.clone(), reduced);
    }
    Ok(Cut {
        data: DataArrayBase::from_parts(da.name().cloned(), var, kept),
        window_dims: pairs.iter().map(|(_, w)| w.clone()).collect(),
        windows: pairs,
        coords,
        split_coords,
        order,
    })
}

impl Coarsen {
    /// The dimension to read lazy data in slabs along, and the slab-length multiple: a
    /// dimension not coarsened if there is one, else the first coarsened one (whole windows).
    fn slab_dim(&self) -> (Name, usize) {
        let coarsened = |d: &Name| self.sizes.iter().any(|(c, _)| c == d);
        match self.source.dims().iter().find(|d| !coarsened(d)) {
            Some(d) => (d.clone(), 1),
            None => self
                .sizes
                .first()
                .map_or((Name::from(""), 1), |(d, w)| (d.clone(), *w)),
        }
    }

    /// The windows as dimensions instead of reducing them: each coarsened dimension `dim`
    /// becomes `(outer, inner)` (blocks, position in the block), coordinates included, as
    /// xarray's `coarsen(...).construct(dim=(outer, inner))`.
    ///
    /// # Errors
    /// [`CoreError::InvalidReduction`] unless every coarsened dimension is named exactly
    /// once; [`CoreError::DuplicateDim`] for names already in use.
    pub fn construct(&self, names: &[(&str, &str, &str)]) -> Result<DataArray> {
        cut(&self.source, &self.sizes, &self.opts, &self.order)?.construct(names)
    }

    /// Reduce every window with `op` (NaN handling, `ddof` and units as for
    /// [`reduce`](DataArrayBase::reduce)). Lazy data is read in slabs of whole windows.
    ///
    /// # Errors
    /// As for [`reduce`](DataArrayBase::reduce).
    pub fn reduce(&self, op: Reduction, opts: &ReduceOptions) -> Result<DataArray> {
        let (dim, multiple) = self.slab_dim();
        crate::slabs::by_slabs(&self.source, &dim, multiple, |s| {
            cut(s, &self.sizes, &self.opts, &self.order)?.reduce(op, opts)
        })
    }
}

impl Cut {
    /// The windows as dimensions instead of reducing them: each coarsened dimension `dim`
    /// becomes `(outer, inner)` (blocks, position in the block), coordinates included, as
    /// xarray's `coarsen(...).construct(dim=(outer, inner))`.
    ///
    /// # Errors
    /// [`CoreError::InvalidReduction`] unless every coarsened dimension is named exactly
    /// once; [`CoreError::DuplicateDim`] for names already in use.
    fn construct(&self, names: &[(&str, &str, &str)]) -> Result<DataArray> {
        for (dim, _) in &self.windows {
            let n = names.iter().filter(|(d, _, _)| *d == dim.as_str()).count();
            if n != 1 {
                return Err(invalid(format!(
                    "construct: give (outer, inner) names for {dim:?} exactly once"
                )));
            }
        }
        if let Some((d, _, _)) = names
            .iter()
            .find(|(d, _, _)| !self.windows.iter().any(|(w, _)| w.as_str() == *d))
        {
            return Err(invalid(format!("construct: {d:?} is not coarsened")));
        }
        let rename = |mut v: Variable| -> Result<Variable> {
            for (dim, outer, inner) in names {
                let window = window_name(&Name::from(*dim));
                if v.dims().iter().any(|d| d.as_str() == *dim) {
                    v = v.rename_dim(dim, &Name::from(*outer))?;
                }
                if v.dims().contains(&window) {
                    v = v.rename_dim(window.as_str(), &Name::from(*inner))?;
                }
            }
            Ok(v)
        };
        let var = rename(self.data.variable().to_owned())?;
        let mut coords: IndexMap<Name, Variable> = IndexMap::new();
        for k in &self.order {
            if let Some(c) = self.data.coords().get(k) {
                coords.insert(k.clone(), c.clone());
            } else if let Some(c) = self.split_coords.get(k) {
                coords.insert(k.clone(), rename(c.clone())?);
            }
        }
        Ok(DataArrayBase::from_parts(
            self.data.name().cloned(),
            var,
            coords,
        ))
    }

    fn reduce(&self, op: Reduction, opts: &ReduceOptions) -> Result<DataArray> {
        let out = self.data.reduce(op, self.window_dims.clone(), opts)?;
        let mut coords: IndexMap<Name, Variable> = out.coords().clone();
        coords.extend(self.coords.iter().map(|(k, c)| (k.clone(), c.clone())));
        let ordered = self
            .order
            .iter()
            .filter_map(|k| coords.get(k).map(|c| (k.clone(), c.clone())))
            .collect();
        Ok(DataArrayBase::from_parts(
            out.name().cloned(),
            out.variable().to_owned(),
            ordered,
        ))
    }
}

macro_rules! shortcuts {
    ($($m:ident $op:ident $doc:literal),* $(,)?) => {
        impl Coarsen {$(
            #[doc = $doc]
            ///
            /// # Errors
            /// As for [`reduce`](Self::reduce).
            pub fn $m(&self) -> Result<DataArray> {
                self.reduce(Reduction::$op, &ReduceOptions::default())
            }
        )*}
    };
}

shortcuts!(
    sum Sum "Sum of each window, skipping NaNs.",
    prod Prod "Product of each window, skipping NaNs.",
    mean Mean "Mean of each window, skipping NaNs.",
    median Median "Median of each window, skipping NaNs.",
    std Std "Population standard deviation of each window, skipping NaNs.",
    var Var "Population variance of each window, skipping NaNs.",
    min Min "Minimum of each window, skipping NaNs.",
    max Max "Maximum of each window, skipping NaNs.",
    count Count "Number of non-NaN values in each window.",
);

/// `(dimension, outer, inner)` names for [`Coarsen::construct`].
type Construct<'a> = (&'a str, &'a str, &'a str);

/// A dataset cut into windows: from [`DatasetBase::coarsen`].
#[derive(Debug, Clone)]
pub struct DatasetCoarsen {
    ds: Dataset,
    windows: Vec<(Name, usize)>,
    opts: CoarsenOptions,
}

impl<S: Repr> DatasetBase<S> {
    /// Cut `windows` into blocks for reduction; see [`DataArrayBase::coarsen`].
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] for a dimension of no variable; [`CoreError::InvalidReduction`]
    /// for a window of 0.
    pub fn coarsen(
        &self,
        windows: &[(&str, usize)],
        opts: &CoarsenOptions,
    ) -> Result<DatasetCoarsen> {
        let sizes = self.sizes();
        for (d, w) in windows {
            if !sizes.contains_key(*d) {
                return Err(CoreError::NoSuchDim(Name::from(*d)));
            }
            if *w == 0 {
                return Err(invalid("a window of 0"));
            }
        }
        Ok(DatasetCoarsen {
            ds: self.to_owned(),
            windows: windows.iter().map(|(d, w)| (Name::from(*d), *w)).collect(),
            opts: *opts,
        })
    }
}

impl DatasetCoarsen {
    /// Reduce every window of every data variable that has a coarsened dimension (others are
    /// kept); non-numeric variables with one are dropped for numeric reductions.
    ///
    /// # Errors
    /// As for [`Coarsen::reduce`].
    pub fn reduce(&self, op: Reduction, opts: &ReduceOptions) -> Result<Dataset> {
        let mut out = Dataset::new();
        let mut coords: IndexMap<Name, Variable> = IndexMap::new();
        let mut vars = Vec::new();
        for k in self.ds.data_vars().keys() {
            let da = self.ds.get(k)?;
            let own: Vec<(&str, usize)> = self
                .windows
                .iter()
                .filter(|(d, _)| da.dims().contains(d))
                .map(|(d, w)| (d.as_str(), *w))
                .collect();
            if own.is_empty() {
                vars.push((k.clone(), da.variable().to_owned()));
                continue;
            }
            if op.numeric_only() && !numeric(da.variable().dtype()) {
                continue;
            }
            let r = da.coarsen(&own, &self.opts)?.reduce(op, opts)?;
            for (ck, c) in r.coords() {
                coords.entry(ck.clone()).or_insert_with(|| c.clone());
            }
            vars.push((k.clone(), r.variable().to_owned()));
        }
        let coarsened: Vec<&Name> = self.windows.iter().map(|(d, _)| d).collect();
        for (k, c) in self.ds.coords() {
            if coords.contains_key(k) {
                continue;
            }
            if c.dims().iter().any(|d| coarsened.contains(&d)) {
                // A coordinate along a coarsened dimension that no variable carried.
                let da = DataArrayBase::from_parts(None, c.clone(), IndexMap::new());
                let own: Vec<(&str, usize)> = self
                    .windows
                    .iter()
                    .filter(|(d, _)| c.dims().contains(d))
                    .map(|(d, w)| (d.as_str(), *w))
                    .collect();
                let r = da.coarsen(&own, &self.opts)?;
                let reduced = if numeric(c.dtype()) {
                    r.reduce(self.opts.coord_reduction, &ReduceOptions::default())?
                } else {
                    r.reduce(Reduction::Min, &ReduceOptions::default())?
                };
                coords.insert(k.clone(), reduced.variable().to_owned());
            } else {
                coords.insert(k.clone(), c.clone());
            }
        }
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in vars {
            out.add_var(k, v)?;
        }
        *out.attrs_mut() = self.ds.attrs().clone();
        Ok(out)
    }

    /// The windows as dimensions for every variable; see [`Coarsen::construct`].
    ///
    /// # Errors
    /// As for [`Coarsen::construct`].
    pub fn construct(&self, names: &[(&str, &str, &str)]) -> Result<Dataset> {
        let mut out = Dataset::new();
        let mut coords: IndexMap<Name, Variable> = IndexMap::new();
        let mut vars = Vec::new();
        let own = |dims: &[Name]| -> (Vec<(&str, usize)>, Vec<Construct<'_>>) {
            let w: Vec<(&str, usize)> = self
                .windows
                .iter()
                .filter(|(d, _)| dims.contains(d))
                .map(|(d, w)| (d.as_str(), *w))
                .collect();
            let n = names
                .iter()
                .filter(|(d, _, _)| w.iter().any(|(x, _)| x == d))
                .copied()
                .collect();
            (w, n)
        };
        for (d, _) in &self.windows {
            if !names.iter().any(|(n, _, _)| *n == d.as_str()) {
                return Err(invalid(format!(
                    "construct: give (outer, inner) names for {d:?}"
                )));
            }
        }
        for k in self.ds.data_vars().keys() {
            let da = self.ds.get(k)?;
            let (w, n) = own(da.dims());
            if w.is_empty() {
                vars.push((k.clone(), da.variable().to_owned()));
                continue;
            }
            let r = da.coarsen(&w, &self.opts)?.construct(&n)?;
            for (ck, c) in r.coords() {
                coords.entry(ck.clone()).or_insert_with(|| c.clone());
            }
            vars.push((k.clone(), r.variable().to_owned()));
        }
        for (k, c) in self.ds.coords() {
            if coords.contains_key(k) {
                continue;
            }
            let (w, n) = own(c.dims());
            if w.is_empty() {
                coords.insert(k.clone(), c.clone());
            } else {
                let da = DataArrayBase::from_parts(None, c.clone(), IndexMap::new());
                let r = da.coarsen(&w, &self.opts)?.construct(&n)?;
                coords.insert(k.clone(), r.variable().to_owned());
            }
        }
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in vars {
            out.add_var(k, v)?;
        }
        *out.attrs_mut() = self.ds.attrs().clone();
        Ok(out)
    }

    /// The mean of each window, skipping NaNs.
    ///
    /// # Errors
    /// As for [`reduce`](Self::reduce).
    pub fn mean(&self) -> Result<Dataset> {
        self.reduce(Reduction::Mean, &ReduceOptions::default())
    }

    /// The sum of each window, skipping NaNs.
    ///
    /// # Errors
    /// As for [`reduce`](Self::reduce).
    pub fn sum(&self) -> Result<Dataset> {
        self.reduce(Reduction::Sum, &ReduceOptions::default())
    }

    /// The maximum of each window, skipping NaNs.
    ///
    /// # Errors
    /// As for [`reduce`](Self::reduce).
    pub fn max(&self) -> Result<Dataset> {
        self.reduce(Reduction::Max, &ReduceOptions::default())
    }

    /// The minimum of each window, skipping NaNs.
    ///
    /// # Errors
    /// As for [`reduce`](Self::reduce).
    pub fn min(&self) -> Result<Dataset> {
        self.reduce(Reduction::Min, &ReduceOptions::default())
    }
}
