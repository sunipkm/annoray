//! Core data model for xarray-rs.
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
//! No function in this crate panics on bad input: all failures are returned as [`CoreError`].

mod array;
mod attrs;
mod dataarray;
mod dataset;
mod display;
mod encoding;
mod error;
mod indexer;
mod label;
mod lazy;
mod lookup;
mod name;
mod ops;
mod repr;
mod select;
mod variable;

pub use array::{Arr, DType, Element, TypedArray};
pub use attrs::{AttrValue, Attrs};
pub use dataarray::{Coords, DataArray, DataArrayBase, DataArrayView, DataArrayViewMut};
pub use dataset::{Dataset, DatasetBase, DatasetView, DatasetViewMut};
pub use encoding::{DatasetEncoding, Encoding};
pub use error::{CoreError, IndexError, Result};
pub use indexer::{Indexer, IntoIndexer, PySlice, SliceIndices, normalize_index};
pub use label::{IntoLabelIndexer, Label, LabelIndexer, Method};
pub use lazy::{LazyArray, LazySource};
pub use name::{Dims, Name};
pub use ops::{BinOp, variables_equal};
pub use repr::{Owned, Repr, ReprMut, View, ViewMut};
pub use variable::{Variable, VariableBase, VariableView, VariableViewMut};
pub use xr_units;

pub use ndarray;

/// Runtime support for the `xr-macros` procedural macros. Not public API.
#[doc(hidden)]
pub mod macro_support {
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
