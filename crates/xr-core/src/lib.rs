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
mod error;
mod indexer;
mod name;
mod repr;
mod variable;

pub use array::{Arr, DType, Element, TypedArray};
pub use attrs::{AttrValue, Attrs};
pub use error::{CoreError, IndexError, Result};
pub use indexer::{Indexer, IntoIndexer, PySlice, SliceIndices, normalize_index};
pub use name::{Dims, Name};
pub use repr::{Owned, Repr, ReprMut, View, ViewMut};
pub use variable::{Variable, VariableBase, VariableView, VariableViewMut};

pub use ndarray;

/// Runtime support for the `xr-macros` procedural macros. Not public API.
#[doc(hidden)]
pub mod macro_support {
    use crate::{
        CoreError, IndexError, Indexer, IntoIndexer, PySlice, Repr, ReprMut, VariableBase,
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

    impl<S: Repr> Isel for VariableBase<S> {
        type View<'a>
            = crate::VariableView<'a>
        where
            Self: 'a;
        fn __xr_isel_pos(&self, items: Vec<Indexer>) -> Result<Self::View<'_>, CoreError> {
            self.isel_pos(items)
        }
        fn __xr_isel_named(
            &self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::View<'_>, CoreError> {
            self.isel(items)
        }
    }

    impl<S: ReprMut> IselMut for VariableBase<S> {
        type ViewMut<'a>
            = crate::VariableViewMut<'a>
        where
            Self: 'a;
        fn __xr_isel_pos_mut(
            &mut self,
            items: Vec<Indexer>,
        ) -> Result<Self::ViewMut<'_>, CoreError> {
            self.isel_pos_mut(items)
        }
        fn __xr_isel_named_mut(
            &mut self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::ViewMut<'_>, CoreError> {
            self.isel_mut(items)
        }
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
