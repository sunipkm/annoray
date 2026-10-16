//! annoray: labelled, units-aware n-dimensional arrays for Rust, with netCDF and HDF5 I/O.
//!
//! Containers come in three storage representations, mirroring ndarray:
//!
//! | owned | shared view | mutable view |
//! |---|---|---|
//! | [`Variable`] | [`VariableView`] | [`VariableViewMut`] |
//!
//! All are aliases of one generic type parameterised by a [`Repr`]. Selecting with
//! [`isel`](VariableBase::isel) returns a view; the `_mut` variants return mutable views whose
//! writes land in the owner.
//!
//! No function in this crate panics on bad input: all failures are returned as errors.
//!
//! Units come from [`units`] (the `annoray-units` crate) and the pure-Rust HDF5 reader and
//! writer from [`hdf5`] (`annoray-hdf5`); both are usable on their own. Files are read and
//! written through [`io`]; [`cf`] applies the CF conventions.
//!
//! ```
//! use annoray::{Variable, isel, ndarray::{ArrayD, IxDyn}};
//!
//! # fn main() -> annoray::Result<()> {
//! let data = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |ix| (ix[0] * 100 + ix[1] * 10 + ix[2]) as i32);
//! let mut v = Variable::new(["time", "lat", "lon"], data)?;
//!
//! // Python-style slices, by name or by position. Results are views.
//! let last = isel!(v[time = -1, lon = ::-2])?;
//! assert_eq!(last.dims(), ["lat", "lon"]);
//! assert_eq!(last.data::<i32>()?[[0, 0]], 103);
//!
//! // Mutable views write through to the owner.
//! isel!(mut v[0, .., 1:3])?.data_mut::<i32>()?.fill(-1);
//! assert_eq!(v.data::<i32>()?[[0, 2, 1]], -1);
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

// The README's examples run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod align;
mod array;
mod assign;
mod attrs;
mod broadcast;
mod calculus;
mod cast;
pub mod cf;
mod choose;
mod coarsen;
mod combine;
mod compare;
mod corr;
mod dataarray;
mod dataset;
mod dataset_ops;
mod display;
mod dt;
mod elementwise;
mod encoding;
mod equality;
mod error;
mod fill;
mod groupby;
mod indexer;
mod interp;
pub mod io;
mod label;
mod lazy;
mod lookup;
mod math;
mod missing;
mod name;
mod nat;
mod ops;
mod reduce;
mod rename;
mod repr;
pub mod resample;
mod reshape;
pub use resample::date_range;
mod rolling;
mod select;
mod slabs;
mod stack;
mod stats;
mod tree;
mod typed;
mod variable;
mod vindex;

pub use align::{Join, align, align_datasets};
pub use annoray_hdf5::{BloscCodec, BloscShuffle, Compression};
pub use array::{Arr, DType, Element, TypedArray};
pub use attrs::{AttrValue, Attrs};
pub use broadcast::broadcast;
pub use calculus::{Covariance, PolyfitOptions, polyval, polyval_dataset};
pub use choose::where_;
pub use coarsen::{Boundary, Coarsen, CoarsenOptions, DatasetCoarsen, Side};
pub use combine::{
    Compat, Concat, ConcatOptions, MergeOptions, combine_by_coords, concat, concat_datasets, merge,
};
pub use compare::{CmpOp, Comparand, LogicOp};
pub use corr::{corr, cov};
pub use dataarray::{Coords, DataArray, DataArrayBase, DataArrayView, DataArrayViewMut};
pub use dataset::{Dataset, DatasetBase, DatasetView, DatasetViewMut};
pub use dt::DtComponent;
pub use elementwise::{ClipBound, dot};
pub use encoding::{DatasetEncoding, Encoding};
pub use error::{CoreError, IndexError, Result};
pub use fill::{FillMethod, InterpNaOptions};
pub use groupby::{BinOptions, DataArrayGroupBy, DatasetGroupBy};
pub use indexer::{Indexer, IntoIndexer, PySlice, SliceIndices, normalize_index};
pub use interp::{Extrapolate, InterpMethod, InterpOptions};
pub use label::{IntoLabelIndexer, Label, LabelIndexer, Method};
pub use lazy::{LazyArray, LazySource, MemorySource};
pub use math::MathFn;
pub use missing::How;
pub use name::{Dims, Name};
pub use nat::NaT;
pub use ops::{BinOp, variables_equal};
pub use reduce::{ReduceDims, ReduceOptions, Reduction, set_stream_bytes, stream_bytes};
pub use repr::{Owned, Repr, ReprMut, View, ViewMut};
pub use rolling::{DataArrayRolling, DatasetRolling, RollingOptions};
pub use stats::{DatasetWeighted, QuantileMethod, Weighted};
pub use tree::DataTree;
pub use variable::{Variable, VariableBase, VariableView, VariableViewMut};

pub use ndarray;

/// Runtime support for the `annoray-macros` procedural macros. Not public API.
#[doc(hidden)]
pub mod __rt {
    use crate::{
        CoreError, IndexError, Indexer, IntoIndexer, IntoLabelIndexer, Label, LabelIndexer, Method,
        PySlice, Repr, ReprMut,
    };

    /// A single indexer expression, e.g. `0`, `1..3`, `PySlice::FULL`.
    pub fn ix<T: IntoIndexer>(v: T) -> Result<Indexer, IndexError> {
        v.into_indexer()
    }

    /// One bound of a `start:stop:step` slice.
    pub fn bound<T: TryInto<isize>>(v: T) -> Result<Option<isize>, IndexError> {
        v.try_into().map(Some).map_err(|_| IndexError::Overflow)
    }

    /// An omitted slice bound.
    pub const fn omitted() -> Result<Option<isize>, IndexError> {
        Ok(None)
    }

    /// A `start:stop:step` slice.
    pub fn slice(
        start: Result<Option<isize>, IndexError>,
        stop: Result<Option<isize>, IndexError>,
        step: Result<Option<isize>, IndexError>,
    ) -> Result<Indexer, IndexError> {
        Ok(Indexer::Slice(PySlice::new(start?, stop?, step?)))
    }

    /// `..`
    pub const fn ellipsis() -> Result<Indexer, IndexError> {
        Ok(Indexer::Ellipsis)
    }

    /// Shared selection for `isel!`. Called with method syntax so receivers auto-deref
    /// (`Variable`, `&Variable`, `&mut Variable`, `Box<Variable>`, … all work).
    pub trait Isel {
        /// The resulting view type.
        type View<'a>
        where
            Self: 'a;
        /// Positional selection.
        fn __xr_isel_pos(&self, items: Vec<Indexer>) -> Result<Self::View<'_>, CoreError>;
        /// Named selection.
        fn __xr_isel_named(
            &self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::View<'_>, CoreError>;
    }

    /// Mutable selection for `isel!(mut ...)`.
    pub trait IselMut {
        /// The resulting mutable view type.
        type ViewMut<'a>
        where
            Self: 'a;
        /// Positional selection.
        fn __xr_isel_pos_mut(
            &mut self,
            items: Vec<Indexer>,
        ) -> Result<Self::ViewMut<'_>, CoreError>;
        /// Named selection.
        fn __xr_isel_named_mut(
            &mut self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::ViewMut<'_>, CoreError>;
    }

    macro_rules! impl_isel {
        ($ty:ident, $view:ident, $view_mut:ident) => {
            impl<S: Repr> Isel for crate::$ty<S> {
                type View<'a>
                    = crate::$view<'a>
                where
                    Self: 'a;
                fn __xr_isel_pos(&self, items: Vec<Indexer>) -> Result<Self::View<'_>, CoreError> {
                    self.view().isel_pos_move(items)
                }
                fn __xr_isel_named(
                    &self,
                    items: Vec<(&'static str, Indexer)>,
                ) -> Result<Self::View<'_>, CoreError> {
                    self.view().isel_move(items)
                }
            }

            impl<S: ReprMut> IselMut for crate::$ty<S> {
                type ViewMut<'a>
                    = crate::$view_mut<'a>
                where
                    Self: 'a;
                fn __xr_isel_pos_mut(
                    &mut self,
                    items: Vec<Indexer>,
                ) -> Result<Self::ViewMut<'_>, CoreError> {
                    self.view_mut().isel_pos_move(items)
                }
                fn __xr_isel_named_mut(
                    &mut self,
                    items: Vec<(&'static str, Indexer)>,
                ) -> Result<Self::ViewMut<'_>, CoreError> {
                    self.view_mut().isel_move(items)
                }
            }
        };
    }

    impl_isel!(VariableBase, VariableView, VariableViewMut);
    impl_isel!(DataArrayBase, DataArrayView, DataArrayViewMut);
    impl_isel!(DatasetBase, DatasetView, DatasetViewMut);

    /// Label selection for `sel!`.
    pub trait Sel {
        /// The resulting view type.
        type View<'a>
        where
            Self: 'a;
        /// Named label selection.
        fn __xr_sel(
            &self,
            items: Vec<(&'static str, LabelIndexer)>,
        ) -> Result<Self::View<'_>, CoreError>;
    }

    /// Mutable label selection for `sel!(mut ...)`.
    pub trait SelMut {
        /// The resulting mutable view type.
        type ViewMut<'a>
        where
            Self: 'a;
        /// Named label selection.
        fn __xr_sel_mut(
            &mut self,
            items: Vec<(&'static str, LabelIndexer)>,
        ) -> Result<Self::ViewMut<'_>, CoreError>;
    }

    macro_rules! impl_sel {
        ($ty:ident, $view:ident, $view_mut:ident) => {
            impl<S: Repr> Sel for crate::$ty<S> {
                type View<'a>
                    = crate::$view<'a>
                where
                    Self: 'a;
                fn __xr_sel(
                    &self,
                    items: Vec<(&'static str, LabelIndexer)>,
                ) -> Result<Self::View<'_>, CoreError> {
                    self.view().sel_move(items)
                }
            }

            impl<S: ReprMut> SelMut for crate::$ty<S> {
                type ViewMut<'a>
                    = crate::$view_mut<'a>
                where
                    Self: 'a;
                fn __xr_sel_mut(
                    &mut self,
                    items: Vec<(&'static str, LabelIndexer)>,
                ) -> Result<Self::ViewMut<'_>, CoreError> {
                    self.view_mut().sel_move(items)
                }
            }
        };
    }

    impl_sel!(DataArrayBase, DataArrayView, DataArrayViewMut);
    impl_sel!(DatasetBase, DatasetView, DatasetViewMut);

    /// A scalar label or label indexer expression (`45.0`, `"2020-01"`, `a..=b`, …).
    pub fn label<T: IntoLabelIndexer>(v: T) -> Result<LabelIndexer, CoreError> {
        v.into_label_indexer()
    }

    /// One bound of a label slice.
    pub fn some_label<T: Into<Label>>(v: T) -> Result<Option<Label>, CoreError> {
        Ok(Some(v.into()))
    }

    /// An omitted label-slice bound.
    pub const fn no_label() -> Result<Option<Label>, CoreError> {
        Ok(None)
    }

    /// A `start:stop:step` label slice (inclusive at both ends).
    pub fn label_slice(
        start: Result<Option<Label>, CoreError>,
        stop: Result<Option<Label>, CoreError>,
        step: Result<Option<isize>, IndexError>,
    ) -> Result<LabelIndexer, CoreError> {
        Ok(LabelIndexer::Slice {
            start: start?,
            stop: stop?,
            step: step?,
        })
    }

    /// `nearest(x)`, `pad(x)`, `backfill(x)`, with an optional tolerance.
    pub fn method<T: Into<Label>>(
        method: Method,
        v: T,
        tolerance: Option<Label>,
    ) -> Result<LabelIndexer, CoreError> {
        Ok(LabelIndexer::Scalar {
            label: v.into(),
            method,
            tolerance,
        })
    }

    /// Re-exported for generated code.
    pub use crate::{Label as LabelType, Method as MethodType};

    /// Collect named label indexers, stopping at the first error.
    pub fn collect_labels<const N: usize>(
        items: [(&'static str, Result<LabelIndexer, CoreError>); N],
    ) -> Result<Vec<(&'static str, LabelIndexer)>, CoreError> {
        items
            .into_iter()
            .map(|(name, ix)| ix.map(|ix| (name, ix)))
            .collect()
    }

    /// Collect positional indexers, stopping at the first error.
    pub fn collect_pos<const N: usize>(
        items: [Result<Indexer, IndexError>; N],
    ) -> Result<Vec<Indexer>, CoreError> {
        Ok(items.into_iter().collect::<Result<Vec<_>, _>>()?)
    }

    /// Collect named indexers, stopping at the first error.
    pub fn collect_named<const N: usize>(
        items: [(&'static str, Result<Indexer, IndexError>); N],
    ) -> Result<Vec<(&'static str, Indexer)>, CoreError> {
        Ok(items
            .into_iter()
            .map(|(name, ix)| ix.map(|ix| (name, ix)))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// The pure-Rust HDF5 reader and writer (the `annoray-hdf5` crate).
pub use annoray_hdf5 as hdf5;
pub use annoray_macros::{isel, sel};
/// Runtime units, CF calendars and dates (the `annoray-units` crate).
pub use annoray_units as units;
pub use annoray_units::{
    Calendar, CfDate, Dyn, Quantity, StaticUnit, Unit, UnitError, UnitPolicy, dimension, dims,
};

/// Any error from annoray.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Data-model errors (dimensions, indexing, arithmetic).
    #[error(transparent)]
    Core(#[from] CoreError),
    /// Unit and calendar errors.
    #[error(transparent)]
    Unit(#[from] UnitError),
    /// File I/O and format errors.
    #[error(transparent)]
    Io(#[from] io::IoError),
    /// CF decoding and encoding errors.
    #[error(transparent)]
    Cf(#[from] cf::CfError),
}
