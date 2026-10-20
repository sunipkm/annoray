//! `put`: a copy of a data array with the values at chosen positions replaced.
//!
//! annoray's values are immutable from Python: an object's data never changes underneath a
//! reference to it. `put` is the update: it returns a new array and leaves the original as it
//! was, so there is no question of which objects a write reaches (xarray's `da[...] = v`
//! reaches views but not copies, depending on how they were selected).
//!
//! - **Where.** Per dimension, one position (the dimension is not part of the selection) or
//!   a list of positions (orthogonal, as `isel` with arrays); dimensions not named are taken
//!   whole. A position may appear once per dimension: repeating it would leave the result
//!   depending on write order.
//! - **Shape.** `value`'s dimensions are a subset of the selection's, each of the selection's
//!   length; missing dimensions are broadcast. A dimension coordinate of `value` along a
//!   selected dimension must equal the target's coordinate at the selected positions.
//! - **Units.** `value` is converted into the target's unit. A unitless value stands beside
//!   a unit only as the scalar NaN, or 0 for a ratio-scale or logarithmic unit (the `where`
//!   rule).
//! - **Element type.** The target's type never changes. Floats are rounded into a narrower
//!   float; integers must fit the target integer type; a float goes into an integer array
//!   only when it is an exact integer. Booleans, text, dates and durations go only into
//!   arrays of their own type.
//! - **Cost.** One copy of the target (lazy data is read without being cached); the write is
//!   a strided assignment when each list of positions is evenly spaced.

use annoray_units::{Unit, UnitKind};
use indexmap::IndexMap;
use ndarray::{ArrayD, ArrayViewD, Axis, Dimension, IxDyn, Slice};

use crate::ops::broadcast_to_dims;
use crate::{
    CoreError, DType, DataArray, DataArrayBase, Name, Owned, Repr, Result, TypedArray, Variable,
    VariableBase, View,
};

/// Where [`put`](DataArrayBase::put) writes along one dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Positions {
    /// One position; the dimension is not part of the selection (`value` must not have it).
    At(usize),
    /// These positions, in this order (the dimension is selected with this length).
    Each(Vec<usize>),
}

fn bad(msg: impl Into<String>) -> CoreError {
    CoreError::Put(msg.into())
}

/// One target axis: whole, a single position, an evenly spaced run, or scattered positions.
enum Plan {
    Whole,
    At(usize),
    Run(Slice),
    Scattered(Vec<usize>),
}

/// Evenly spaced positions as a slice (any nonzero step, as ndarray's).
fn run(p: &[usize]) -> Option<Slice> {
    let (&first, rest) = p.split_first()?;
    let Some(&second) = rest.first() else {
        let end = isize::try_from(first).ok()?.checked_add(1)?;
        return Some(Slice::new(isize::try_from(first).ok()?, Some(end), 1));
    };
    let step = isize::try_from(second)
        .ok()?
        .checked_sub(isize::try_from(first).ok()?)?;
    if step == 0 {
        return None;
    }
    for w in p.windows(2) {
        if let [a, b] = w
            && isize::try_from(*b)
                .ok()?
                .checked_sub(isize::try_from(*a).ok()?)?
                != step
        {
            return None;
        }
    }
    let last = *p.last()?;
    let (lo, hi) = (first.min(last), first.max(last));
    let lo = isize::try_from(lo).ok()?;
    let hi = isize::try_from(hi).ok()?.checked_add(1)?;
    Some(Slice::new(lo, Some(hi), step))
}

/// `true` when unitless `bare` may stand beside data in `unit` (the `where` rule).
fn bare_ok<R: Repr>(unit: &Unit, bare: &VariableBase<R>) -> bool {
    let scalar = |f: &dyn Fn(f64) -> bool| {
        bare.ndim() == 0 && !bare.is_lazy() && bare.to_f64().is_ok_and(|a| a.iter().all(|x| f(*x)))
    };
    let zero_ok = matches!(unit.kind(), UnitKind::Linear | UnitKind::Log(_));
    scalar(&|x: f64| x.is_nan()) || (zero_ok && scalar(&|x: f64| x == 0.0))
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

fn is_float(d: DType) -> bool {
    matches!(d, DType::F32 | DType::F64)
}

/// `v` as values of `target`, under the module's rules.
fn fit(v: &TypedArray<View<'_>>, target: DType) -> Result<TypedArray<Owned>> {
    let from = v.dtype();
    if from == target {
        return Ok(v.to_owned());
    }
    let mismatch = || CoreError::DTypeMismatch {
        expected: target,
        found: from,
    };
    if is_float(target) && (is_int(from) || is_float(from)) {
        return crate::cast::cast(v, target);
    }
    if is_int(target) && is_int(from) {
        return crate::cast::cast(v, target);
    }
    if is_int(target) && is_float(from) {
        let f = crate::cast::cast(v, DType::F64)?;
        let TypedArray::F64(a) = &f else {
            return Err(mismatch());
        };
        if let Some(x) = a.iter().find(|x| x.fract() != 0.0 || !x.is_finite()) {
            return Err(CoreError::Cast {
                value: x.to_string(),
                dtype: target,
                reason: "not an exact integer (round it first)",
            });
        }
        return crate::cast::cast(&f.view(), target);
    }
    Err(mismatch())
}

/// Write `src` (already in the selection's dimension order, single-position axes absent)
/// into `dst` at `plans`.
fn write<T: Clone>(dst: &mut ArrayD<T>, plans: &[Plan], src: ArrayViewD<'_, T>) -> Result<()> {
    // Single-position axes become length 1 in the source, so the shapes line up.
    let mut src = src;
    for (axis, p) in plans.iter().enumerate() {
        if matches!(p, Plan::At(_)) {
            src = src.insert_axis(Axis(axis));
        }
    }
    if !plans.iter().any(|p| matches!(p, Plan::Scattered(_))) {
        let mut view = dst.slice_each_axis_mut(|ax| match plans.get(ax.axis.index()) {
            Some(Plan::At(p)) => Slice::from(*p..p.saturating_add(1)),
            Some(Plan::Run(s)) => *s,
            _ => Slice::from(..),
        });
        let src = src
            .broadcast(view.raw_dim())
            .ok_or_else(|| bad("value does not broadcast to the selection"))?;
        view.assign(&src);
        return Ok(());
    }
    // Scattered positions: element by element over the selection.
    let positions: Vec<Vec<usize>> = plans
        .iter()
        .zip(dst.shape())
        .map(|(p, n)| match p {
            Plan::Whole => (0..*n).collect(),
            Plan::At(i) => vec![*i],
            Plan::Run(s) => {
                let len = isize::try_from(*n).unwrap_or(isize::MAX);
                let start = usize::try_from(s.start).unwrap_or(0);
                let end = s
                    .end
                    .and_then(|e| usize::try_from(e.min(len)).ok())
                    .unwrap_or(*n);
                let r: Vec<usize> = (start..end).collect();
                if s.step < 0 {
                    r.into_iter().rev().step_by(s.step.unsigned_abs()).collect()
                } else {
                    r.into_iter().step_by(s.step.unsigned_abs()).collect()
                }
            }
            Plan::Scattered(v) => v.clone(),
        })
        .collect();
    let shape: Vec<usize> = positions.iter().map(Vec::len).collect();
    let src = src
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| bad("value does not broadcast to the selection"))?;
    let mut at = vec![0usize; shape.len()];
    for (ix, v) in src.indexed_iter() {
        for ((slot, i), p) in at.iter_mut().zip(ix.slice()).zip(&positions) {
            *slot = p.get(*i).copied().unwrap_or(0);
        }
        if let Some(d) = dst.get_mut(IxDyn(&at)) {
            *d = v.clone();
        }
    }
    Ok(())
}

impl<S: Repr> DataArrayBase<S> {
    /// A copy with the values at `at` replaced by `value`; name, coordinates, attributes and
    /// encoding are kept and `self` is unchanged.
    ///
    /// - **Where:** per dimension, one position ([`Positions::At`]; the dimension leaves the
    ///   selection) or a list ([`Positions::Each`], orthogonal as `isel` with arrays);
    ///   dimensions not named are taken whole. A position may appear once per dimension.
    /// - **Shape:** `value`'s dimensions are a subset of the selection's, each of the
    ///   selection's length; missing dimensions are broadcast. A dimension coordinate of
    ///   `value` along a selected dimension must equal `self`'s at the selected positions.
    /// - **Units:** `value` is converted into `self`'s unit; a unitless value stands beside a
    ///   unit only as the scalar NaN, or 0 for a ratio-scale or logarithmic unit.
    /// - **Element type:** never changes. Floats round into a narrower float; integers must
    ///   fit; a float goes into an integer array only as an exact integer; booleans, text,
    ///   dates and durations only into arrays of their own type.
    /// - **Cost:** one copy of `self` (lazy data is read without being cached); evenly spaced
    ///   positions are written by a strided assignment.
    ///
    /// ```
    /// use annoray::{DataArray, Name, Positions, Variable};
    /// use ndarray::{ArrayD, IxDyn, array};
    ///
    /// # fn main() -> annoray::Result<()> {
    /// let a = DataArray::new(Variable::new(["x"], array![1.0, 2.0, 3.0, 4.0].into_dyn())?);
    /// let zero = DataArray::new(Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), 0.0))?);
    /// let b = a.put(&[(Name::from("x"), Positions::Each(vec![0, 2]))], &zero)?;
    /// assert_eq!(b.variable().data::<f64>()?.as_slice(), Some(&[0.0, 2.0, 0.0, 4.0][..]));
    /// assert_eq!(a.variable().data::<f64>()?[0], 1.0); // unchanged
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::Put`] for positions out of range or repeated,
    /// or a `value` whose dimensions or coordinates do not fit the selection; unit errors;
    /// [`CoreError::DTypeMismatch`] or [`CoreError::Cast`] when `value` does not fit the
    /// element type; read errors for lazy data.
    pub fn put<R: Repr>(
        &self,
        at: &[(Name, Positions)],
        value: &DataArrayBase<R>,
    ) -> Result<DataArray> {
        let dims = self.dims();
        let shape = self.shape();
        let mut chosen: IndexMap<&Name, &Positions> = IndexMap::new();
        for (d, p) in at {
            let axis = dims
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| CoreError::NoSuchDim(d.clone()))?;
            let n = shape.get(axis).copied().unwrap_or(0);
            if chosen.insert(d, p).is_some() {
                return Err(bad(format!("dimension {d:?} given twice")));
            }
            let list: &[usize] = match p {
                Positions::At(i) => std::slice::from_ref(i),
                Positions::Each(v) => v,
            };
            if let Some(i) = list.iter().find(|i| **i >= n) {
                return Err(bad(format!(
                    "position {i} is out of range for dimension {d:?} of length {n}"
                )));
            }
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w.first() == w.get(1)) {
                return Err(bad(format!(
                    "a position along {d:?} is given twice (the result would depend on \
                     write order)"
                )));
            }
        }

        // The selection: its dimensions and lengths, in this array's order.
        let mut sel_dims: Vec<Name> = Vec::new();
        let mut sel_len: Vec<usize> = Vec::new();
        let mut plans: Vec<Plan> = Vec::new();
        for (d, n) in dims.iter().zip(shape) {
            match chosen.get(d) {
                None => {
                    sel_dims.push(d.clone());
                    sel_len.push(*n);
                    plans.push(Plan::Whole);
                }
                Some(Positions::At(i)) => plans.push(Plan::At(*i)),
                Some(Positions::Each(v)) => {
                    sel_dims.push(d.clone());
                    sel_len.push(v.len());
                    plans.push(run(v).map_or_else(|| Plan::Scattered(v.clone()), Plan::Run));
                }
            }
        }
        for (d, n) in value.dims().iter().zip(value.shape()) {
            match sel_dims.iter().position(|x| x == d) {
                None => {
                    return Err(bad(format!(
                        "value has dimension {d:?}, which the selection does not"
                    )));
                }
                Some(k) if sel_len.get(k) != Some(n) => {
                    return Err(CoreError::DimLengthMismatch {
                        dim: d.clone(),
                        expected: sel_len.get(k).copied().unwrap_or(0),
                        found: *n,
                    });
                }
                Some(_) => {}
            }
        }
        // Dimension coordinates of the value must label the same places.
        for (k, c) in value.coords() {
            if c.dims() != [k.clone()] || !sel_dims.contains(k) {
                continue;
            }
            let Some(mine) = self.coords().get(k) else {
                continue;
            };
            let picked = match chosen.get(k) {
                Some(Positions::Each(v)) => {
                    let idx: Vec<isize> = v
                        .iter()
                        .map(|i| isize::try_from(*i).map_err(|_| crate::IndexError::Overflow))
                        .collect::<std::result::Result<_, _>>()?;
                    mine.take(k, &idx)?
                }
                _ => mine.to_owned(),
            };
            if !crate::equality::same_values(&picked, c, false)? {
                return Err(CoreError::NotAligned(k.clone()));
            }
        }

        // Units: the value in this array's unit.
        let value = match (self.unit(), value.unit()) {
            (None, None) => value.to_owned(),
            (Some(a), Some(b)) if a == b => value.to_owned(),
            (Some(a), Some(_)) => value.to_unit(a)?,
            (Some(a), None) if bare_ok(a, value.variable()) => value.to_owned(),
            (None, Some(b)) if bare_ok(b, value.variable()) => value.to_owned(),
            (mine, theirs) => {
                let name = |u: Option<&Unit>| {
                    u.map_or_else(|| Unit::one().to_string(), ToString::to_string)
                };
                return Err(CoreError::Unit(annoray_units::UnitError::Incompatible {
                    lhs: name(mine),
                    rhs: name(theirs),
                }));
            }
        };
        let src = fit(&value.variable().typed()?, self.dtype())?;

        // One copy of the target (lazy data read afresh, not cached), then the write.
        let var = self.variable();
        let mut data = var.to_owned().into_typed()?;
        macro_rules! go {
            ($($v:ident),*) => {
                match (&mut data, &src) {
                    $((TypedArray::$v(d), TypedArray::$v(s)) => {
                        let s = broadcast_to_dims(s.view(), value.dims(), &sel_dims)?;
                        write(d, &plans, s)?;
                    })*
                    _ => return Err(CoreError::DTypeMismatch {
                        expected: self.dtype(),
                        found: src.dtype(),
                    }),
                }
            };
        }
        go!(
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool, Str, DateTime, CfDate, TimeDelta
        );

        let mut out =
            Variable::from_typed(dims.to_vec(), data)?.with_encoding(var.encoding().clone());
        if let Some(u) = var.unit() {
            out = out.with_unit(u.clone());
        }
        out.take_zone_from(var);
        *out.attrs_mut() = var.attrs().clone();
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArray::from_parts(self.name().cloned(), out, coords))
    }
}
