//! Whole-object comparison (xarray's `equals`, `identical`, `broadcast_equals`).
//!
//! - `equals`: the same dimensions (in order), coordinates and values. Values in different
//!   units of a kind are converted (1 km equals 1000 m, with the exactness of comparisons);
//!   missing values (NaN, NaT) equal each other; values of different kinds (text and
//!   numbers) are not equal. Names and attributes are not compared.
//! - `identical`: `equals`, and the same units exactly, names, and attributes (of the
//!   object and of every variable; NaN attribute values equal each other, as in xarray).
//! - `broadcast_equals`: `equals` after broadcasting the two against each other.
//!
//! Lazy data is read.

use crate::{
    AttrValue, Attrs, CmpOp, CoreError, DataArrayBase, DataArrayView, DatasetBase, Repr, Result,
    VariableBase,
};

/// Whether two variables hold the same values (see the module notes); `exact` also requires
/// the same unit.
fn same_values<S: Repr, R: Repr>(
    a: &VariableBase<S>,
    b: &VariableBase<R>,
    exact: bool,
) -> Result<bool> {
    if a.dims() != b.dims() || a.shape() != b.shape() {
        return Ok(false);
    }
    let units_ok = match (a.unit(), b.unit()) {
        (None, None) => true,
        (Some(x), Some(y)) => x == y || (!exact && x.is_convertible(y)),
        _ => false,
    };
    if !units_ok {
        return Ok(false);
    }
    if crate::variables_equal(a, b) {
        return Ok(true);
    }
    let (x, y) = (DataArrayView::from(a.view()), DataArrayView::from(b.view()));
    let eq = match x.compare(CmpOp::Eq, &y) {
        Ok(e) => e,
        Err(CoreError::UnsupportedDType { .. } | CoreError::Unit(_)) => return Ok(false),
        Err(e) => return Err(e),
    };
    let (nx, ny) = (x.isnull()?, y.isnull()?);
    let (eq, nx, ny) = (eq.data::<bool>()?, nx.data::<bool>()?, ny.data::<bool>()?);
    Ok(ndarray::Zip::from(&eq)
        .and(&nx)
        .and(&ny)
        .all(|e, p, q| *e || (*p && *q)))
}

/// Attribute values equal, with NaN equal to NaN.
fn attr_equal(a: &AttrValue, b: &AttrValue) -> bool {
    fn floats<T: num_like::Float>(x: &[T], y: &[T]) -> bool {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p == q || (p.nan() && q.nan()))
    }
    match (a, b) {
        (AttrValue::F64(x), AttrValue::F64(y)) => floats(x, y),
        (AttrValue::F32(x), AttrValue::F32(y)) => floats(x, y),
        _ => a == b,
    }
}

mod num_like {
    pub(super) trait Float: PartialEq {
        fn nan(&self) -> bool;
    }
    impl Float for f64 {
        fn nan(&self) -> bool {
            self.is_nan()
        }
    }
    impl Float for f32 {
        fn nan(&self) -> bool {
            self.is_nan()
        }
    }
}

/// Attribute sets equal (any order; NaN equal to NaN).
pub(crate) fn attrs_equal(a: &Attrs, b: &Attrs) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|(k, v)| b.get(k).is_some_and(|w| attr_equal(v, w)))
}

fn same_variable<S: Repr, R: Repr>(
    a: &VariableBase<S>,
    b: &VariableBase<R>,
    identical: bool,
) -> Result<bool> {
    Ok(same_values(a, b, identical)? && (!identical || attrs_equal(a.attrs(), b.attrs())))
}

/// Coordinates of `a` and `b` (as name → variable maps) match one to one.
fn same_coords<'a, 'b, S: Repr + 'a, R: Repr + 'b>(
    a: impl ExactSizeIterator<Item = (&'a crate::Name, &'a VariableBase<S>)>,
    b: &indexmap::IndexMap<crate::Name, VariableBase<R>>,
    identical: bool,
) -> Result<bool> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (k, v) in a {
        match b.get(k) {
            Some(w) if same_variable(v, w, identical)? => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

impl<S: Repr> DataArrayBase<S> {
    /// Whether `other` has the same dimensions, coordinates and values (see the module
    /// notes).
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn equals<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<bool> {
        Ok(same_values(self.variable(), other.variable(), false)?
            && same_coords(self.coords().iter(), other.coords(), false)?)
    }

    /// [`equals`](Self::equals), and the same units, name and attributes.
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn identical<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<bool> {
        Ok(self.name() == other.name()
            && same_variable(self.variable(), other.variable(), true)?
            && same_coords(self.coords().iter(), other.coords(), true)?)
    }

    /// Whether the two are [equal](Self::equals) once broadcast against each other (the
    /// dimension order of the result of broadcasting).
    ///
    /// # Errors
    /// Read errors for lazy data; alignment errors.
    pub fn broadcast_equals<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<bool> {
        let b = match crate::broadcast(&[&self.to_owned(), &other.to_owned()], &[]) {
            Ok(b) => b,
            Err(CoreError::DimLengthMismatch { .. }) => return Ok(false),
            Err(e) => return Err(e),
        };
        match (b.first(), b.get(1)) {
            (Some(x), Some(y)) => {
                let order: Vec<&str> = x.dims().iter().map(crate::Name::as_str).collect();
                x.equals(&y.transpose(&order)?)
            }
            _ => Ok(false),
        }
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Whether `other` has the same variables (any order), each with the same dimensions
    /// and values (see the module notes).
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn equals<R: Repr>(&self, other: &DatasetBase<R>) -> Result<bool> {
        Ok(
            same_coords(self.data_vars().iter(), other.data_vars(), false)?
                && same_coords(self.coords().iter(), other.coords(), false)?,
        )
    }

    /// [`equals`](Self::equals), and the same units and attributes (global and per variable).
    ///
    /// # Errors
    /// Read errors for lazy data.
    pub fn identical<R: Repr>(&self, other: &DatasetBase<R>) -> Result<bool> {
        Ok(attrs_equal(self.attrs(), other.attrs())
            && same_coords(self.data_vars().iter(), other.data_vars(), true)?
            && same_coords(self.coords().iter(), other.coords(), true)?)
    }
}
