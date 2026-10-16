//! Dataset arithmetic, comparisons and elementwise functions, variable by variable (xarray's
//! dataset operators).
//!
//! - With a number or a quantity: every data variable.
//! - With a data array: aligned first (inner join on labels), then broadcast against every data
//!   variable.
//! - With another dataset: aligned first (inner join), then the data variables both have
//!   (xarray's default `arithmetic_join` and `dataset_join`, both "inner"); variables in one
//!   only are dropped.
//! - Units, dtypes, NaN and laziness as for the data-array operation; a variable the operation
//!   refuses (text in arithmetic, °C in a product) makes the whole operation fail rather than
//!   being dropped silently. Coordinates are kept and merged. Attributes are kept, as xarray's
//!   default `keep_attrs`: variables as their operation leaves them (both operands' attributes,
//!   conflicting keys dropped), the dataset's own (two datasets: both, conflicts dropped). Unlike
//!   xarray, a data-array operand's attributes are not merged into the dataset's.

use crate::align::{Join, Obj, apply_array, apply_dataset, plan};
use crate::{
    BinOp, CmpOp, DataArray, DataArrayBase, Dataset, DatasetBase, LogicOp, MathFn, Repr, Result,
    View,
};
use annoray_units::UnitPolicy;

/// How the attributes of a per-variable result are chosen.
#[derive(Clone, Copy)]
enum Keep {
    /// As the operation produced them (operators, functions).
    Result,
    /// Each variable's own, from the source dataset (`map` with `keep_attrs`).
    Source,
    /// None (`map` with `keep_attrs=False`).
    Drop,
}

/// Collect per-variable results into a dataset with `coords_of`'s coordinates and the given
/// dataset attributes.
fn assemble<S: Repr>(
    coords_of: &DatasetBase<S>,
    results: Vec<(crate::Name, DataArray)>,
    keep: Keep,
    attrs: crate::Attrs,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    for (k, c) in coords_of.coords() {
        out.add_coord(k.clone(), c.to_owned())?;
    }
    for (k, mut r) in results {
        match keep {
            Keep::Result => {}
            Keep::Drop => r.attrs_mut().clear(),
            Keep::Source => {
                if let Ok(src) = coords_of.get(&k) {
                    *r.attrs_mut() = src.attrs().clone();
                }
            }
        }
        out.add_array(k, r)?;
    }
    *out.attrs_mut() = attrs;
    Ok(out)
}

impl<S: Repr> DatasetBase<S> {
    /// `f` of every data variable (xarray's `Dataset.map`). With `keep_attrs` (xarray's
    /// default) the dataset and each variable keep their attributes; without, both are dropped.
    ///
    /// # Errors
    /// The first error `f` returns; coordinate conflicts between results.
    pub fn map_data_vars(
        &self,
        keep_attrs: bool,
        f: impl Fn(&DataArrayBase<View<'_>>) -> Result<DataArray>,
    ) -> Result<Dataset> {
        let mut results = Vec::with_capacity(self.data_vars().len());
        for k in self.data_vars().keys() {
            results.push((k.clone(), f(&self.get(k)?)?));
        }
        if keep_attrs {
            assemble(self, results, Keep::Source, self.attrs().clone())
        } else {
            assemble(self, results, Keep::Drop, crate::Attrs::new())
        }
    }

    /// `f` of every data variable, attributes as the operation leaves them and the dataset's
    /// kept (operators and elementwise functions).
    fn apply(&self, f: impl Fn(&DataArrayBase<View<'_>>) -> Result<DataArray>) -> Result<Dataset> {
        let mut results = Vec::with_capacity(self.data_vars().len());
        for k in self.data_vars().keys() {
            results.push((k.clone(), f(&self.get(k)?)?));
        }
        assemble(self, results, Keep::Result, self.attrs().clone())
    }

    /// `f(variable, other)` for every data variable, after aligning with `other`.
    ///
    /// # Errors
    /// Alignment errors; the first error `f` returns.
    pub fn zip_with_array<R: Repr>(
        &self,
        other: &DataArrayBase<R>,
        f: impl Fn(&DataArrayBase<View<'_>>, &DataArray) -> Result<DataArray>,
    ) -> Result<Dataset> {
        let p = plan(&[Obj::of_dataset(self), Obj::of_array(other)], Join::Inner)?;
        let (ds, da) = if p.is_noop() {
            (self.to_owned(), other.to_owned())
        } else {
            (apply_dataset(self, &p, 0)?, apply_array(other, &p, 1)?)
        };
        let mut results = Vec::with_capacity(ds.data_vars().len());
        for k in ds.data_vars().keys() {
            results.push((k.clone(), f(&ds.get(k)?, &da)?));
        }
        // The dataset's attributes; xarray also merges the data array's variable attributes into
        // them, which mixes levels (not done here).
        let attrs = ds.attrs().clone();
        assemble(&ds, results, Keep::Result, attrs)
    }

    /// `f(variable, its namesake in other)` for the data variables both have, after aligning
    /// the datasets.
    ///
    /// # Errors
    /// Alignment errors; the first error `f` returns.
    pub fn zip_with_dataset<R: Repr>(
        &self,
        other: &DatasetBase<R>,
        f: impl Fn(&DataArrayBase<View<'_>>, &DataArrayBase<View<'_>>) -> Result<DataArray>,
    ) -> Result<Dataset> {
        let p = plan(
            &[Obj::of_dataset(self), Obj::of_dataset(other)],
            Join::Inner,
        )?;
        let (a, b) = if p.is_noop() {
            (self.to_owned(), other.to_owned())
        } else {
            (apply_dataset(self, &p, 0)?, apply_dataset(other, &p, 1)?)
        };
        let mut results = Vec::new();
        for k in a.data_vars().keys() {
            if b.data_vars().contains_key(k) {
                results.push((k.clone(), f(&a.get(k)?, &b.get(k)?)?));
            }
        }
        // Coordinates of both (the aligned left one first).
        let attrs = crate::attrs::drop_conflicts(a.attrs(), b.attrs());
        let mut out = assemble(&a, results, Keep::Result, attrs)?;
        for (k, c) in b.coords() {
            if !out.coords().contains_key(k) {
                out.add_coord(k.clone(), c.clone())?;
            }
        }
        Ok(out)
    }

    /// `self op rhs` for every data variable, `rhs` a data array; see the module notes.
    ///
    /// # Errors
    /// As for [`DataArrayBase::binary`].
    pub fn binary_array<R: Repr>(
        &self,
        op: BinOp,
        rhs: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<Dataset> {
        self.zip_with_array(rhs, |v, o| v.binary(op, o, policy))
    }

    /// `self op rhs` for the data variables both datasets have; see the module notes.
    ///
    /// # Errors
    /// As for [`DataArrayBase::binary`].
    pub fn binary<R: Repr>(
        &self,
        op: BinOp,
        rhs: &DatasetBase<R>,
        policy: UnitPolicy,
    ) -> Result<Dataset> {
        self.zip_with_dataset(rhs, |a, b| a.binary(op, b, policy))
    }

    /// `self op scalar` for every data variable (a bare number: unit rules as for data
    /// arrays).
    ///
    /// # Errors
    /// As for [`DataArrayBase::binary_scalar`].
    pub fn binary_scalar(&self, op: BinOp, scalar: f64) -> Result<Dataset> {
        self.apply(|v| v.binary_scalar(op, scalar))
    }

    /// Elementwise comparison of every data variable with a data array.
    ///
    /// # Errors
    /// As for [`DataArrayBase::compare`].
    pub fn compare_array<R: Repr>(&self, op: CmpOp, rhs: &DataArrayBase<R>) -> Result<Dataset> {
        self.zip_with_array(rhs, |v, o| v.compare(op, o))
    }

    /// Elementwise comparison of the data variables both datasets have.
    ///
    /// # Errors
    /// As for [`DataArrayBase::compare`].
    pub fn compare<R: Repr>(&self, op: CmpOp, rhs: &DatasetBase<R>) -> Result<Dataset> {
        self.zip_with_dataset(rhs, |a, b| a.compare(op, b))
    }

    /// Logical operation of every data variable with a boolean data array.
    ///
    /// # Errors
    /// As for [`DataArrayBase::logical`].
    pub fn logical_array<R: Repr>(&self, op: LogicOp, rhs: &DataArrayBase<R>) -> Result<Dataset> {
        self.zip_with_array(rhs, |v, o| v.logical(op, o))
    }

    /// Logical operation of the data variables both datasets have.
    ///
    /// # Errors
    /// As for [`DataArrayBase::logical`].
    pub fn logical<R: Repr>(&self, op: LogicOp, rhs: &DatasetBase<R>) -> Result<Dataset> {
        self.zip_with_dataset(rhs, |a, b| a.logical(op, b))
    }

    /// Logical not of every (boolean) data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::logical_not`].
    pub fn logical_not(&self) -> Result<Dataset> {
        self.apply(|v| v.logical_not())
    }

    /// [`DataArrayBase::math`] of every data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::math`].
    pub fn math(&self, f: MathFn) -> Result<Dataset> {
        self.apply(|v| v.math(f))
    }

    /// [`DataArrayBase::pow`] of every data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::pow`].
    pub fn pow(&self, e: f64) -> Result<Dataset> {
        self.apply(|v| v.pow(e))
    }

    /// Keep values where `cond` (a boolean data array) holds, NaN (NaT) elsewhere, in every
    /// data variable.
    ///
    /// # Errors
    /// As for [`DataArrayBase::where_`].
    pub fn where_array<R: Repr>(&self, cond: &DataArrayBase<R>) -> Result<Dataset> {
        self.zip_with_array(cond, |v, c| v.where_(c))
    }

    /// Keep values where `cond`'s namesake variable holds (a dataset of booleans).
    ///
    /// # Errors
    /// As for [`DataArrayBase::where_`].
    pub fn where_dataset<R: Repr>(&self, cond: &DatasetBase<R>) -> Result<Dataset> {
        self.zip_with_dataset(cond, |v, c| v.where_(c))
    }

    /// Where every data variable is missing (NaN/NaT).
    ///
    /// # Errors
    /// [`crate::CoreError::Read`] if lazy data cannot be read.
    pub fn isnull(&self) -> Result<Dataset> {
        self.apply(|v| v.isnull())
    }

    /// Where every data variable is present.
    ///
    /// # Errors
    /// As for [`isnull`](Self::isnull).
    pub fn notnull(&self) -> Result<Dataset> {
        self.apply(|v| v.notnull())
    }
}

/// `+ - * /` for datasets, as for data arrays: with a dataset, a data array or a number (on
/// either side for numbers), and with a `Result` on one side.
macro_rules! impl_ds_ops {
    ($($tr:ident $m:ident $op:ident),*) => {$(
        impl<S: Repr, R: Repr> std::ops::$tr<&DatasetBase<R>> for &DatasetBase<S> {
            type Output = Result<Dataset>;
            fn $m(self, rhs: &DatasetBase<R>) -> Result<Dataset> {
                self.binary(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
        impl<R: Repr> std::ops::$tr<&DatasetBase<R>> for Result<Dataset> {
            type Output = Result<Dataset>;
            fn $m(self, rhs: &DatasetBase<R>) -> Result<Dataset> {
                self?.binary(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
        impl<S: Repr, R: Repr> std::ops::$tr<&DataArrayBase<R>> for &DatasetBase<S> {
            type Output = Result<Dataset>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<Dataset> {
                self.binary_array(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
        impl<S: Repr> std::ops::$tr<f64> for &DatasetBase<S> {
            type Output = Result<Dataset>;
            fn $m(self, rhs: f64) -> Result<Dataset> {
                self.binary_scalar(BinOp::$op, rhs)
            }
        }
        impl<S: Repr> std::ops::$tr<&DatasetBase<S>> for f64 {
            type Output = Result<Dataset>;
            fn $m(self, rhs: &DatasetBase<S>) -> Result<Dataset> {
                rhs.apply(|v| std::ops::$tr::$m(self, v))
            }
        }
    )*};
}

impl_ds_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div);

impl<S: Repr> std::ops::Neg for &DatasetBase<S> {
    type Output = Result<Dataset>;
    fn neg(self) -> Result<Dataset> {
        self.math(MathFn::Neg)
    }
}

impl<S: Repr> std::ops::Not for &DatasetBase<S> {
    type Output = Result<Dataset>;
    fn not(self) -> Result<Dataset> {
        self.logical_not()
    }
}
