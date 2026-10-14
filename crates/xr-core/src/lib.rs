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

    /// Something `isel!` can be applied to. Implemented for `&C` (giving a view) and `&mut C`
    /// (giving a mutable view) for each container `C`.
    pub trait Isel {
        /// The resulting view type.
        type Output;
        /// Positional selection.
        fn isel_pos_dyn(self, items: Vec<Indexer>) -> Result<Self::Output, CoreError>;
        /// Named selection.
        fn isel_named_dyn(
            self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::Output, CoreError>;
    }

    impl<'a, S: Repr> Isel for &'a VariableBase<S> {
        type Output = crate::VariableView<'a>;
        fn isel_pos_dyn(self, items: Vec<Indexer>) -> Result<Self::Output, CoreError> {
            self.isel_pos(items)
        }
        fn isel_named_dyn(
            self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::Output, CoreError> {
            self.isel(items)
        }
    }

    impl<'a, S: ReprMut> Isel for &'a mut VariableBase<S> {
        type Output = crate::VariableViewMut<'a>;
        fn isel_pos_dyn(self, items: Vec<Indexer>) -> Result<Self::Output, CoreError> {
            self.isel_pos_mut(items)
        }
        fn isel_named_dyn(
            self,
            items: Vec<(&'static str, Indexer)>,
        ) -> Result<Self::Output, CoreError> {
            self.isel_mut(items)
        }
    }

    /// Entry point for positional `isel!`.
    pub fn isel_pos<T: Isel, const N: usize>(
        target: T,
        items: [Result<Indexer, IndexError>; N],
    ) -> Result<T::Output, CoreError> {
        let items = items.into_iter().collect::<Result<Vec<_>, _>>()?;
        target.isel_pos_dyn(items)
    }

    /// Entry point for named `isel!`.
    pub fn isel_named<T: Isel, const N: usize>(
        target: T,
        items: [(&'static str, Result<Indexer, IndexError>); N],
    ) -> Result<T::Output, CoreError> {
        let items = items
            .into_iter()
            .map(|(name, ix)| ix.map(|ix| (name, ix)))
            .collect::<Result<Vec<_>, _>>()?;
        target.isel_named_dyn(items)
    }
}
