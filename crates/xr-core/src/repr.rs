//! Storage representations: the `S` in `VariableBase<S>` / `DatasetBase<S>`.
//!
//! Like ndarray's `OwnedRepr` / `ViewRepr`, a [`Repr`] decides whether a container owns its
//! buffers, borrows them, or borrows them mutably. It does this for every element type at once
//! through the generic associated type [`Repr::Data`].

use std::borrow::Borrow;
use std::fmt::Debug;
use std::marker::PhantomData;

use ndarray::{ArrayBase, ArrayViewMutD, IxDyn, OwnedRepr, ViewRepr};

use crate::{Attrs, Element};

mod sealed {
    pub trait Sealed {}
}

/// A storage representation for named-dimension containers. Sealed.
pub trait Repr: sealed::Sealed + Sized {
    /// The ndarray storage used for elements of type `T`.
    type Data<T: Element>: ndarray::Data<Elem = T>;
    /// How attributes are held: owned, or borrowed from the owner.
    type Attrs: Borrow<Attrs> + Debug;
    /// How coordinates are held. Mutable views hold coordinates as shared views: only data
    /// variables can be written through them.
    type Coord: Repr<Attrs = Self::Attrs>;
}

/// A representation that allows mutating element data.
pub trait ReprMut: Repr {
    /// Mutably view an array stored in this representation.
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T>;
}

/// Owned storage: `Variable`, `Dataset`, ….
#[derive(Debug)]
pub struct Owned;

/// Shared borrowed storage: `VariableView<'a>`, `DatasetView<'a>`, ….
#[derive(Debug)]
pub struct View<'a>(PhantomData<&'a ()>);

/// Mutably borrowed storage: `VariableViewMut<'a>`, `DatasetViewMut<'a>`, ….
///
/// Element data can be written through it. Dimensions, coordinates and attributes cannot.
#[derive(Debug)]
pub struct ViewMut<'a>(PhantomData<&'a mut ()>);

impl sealed::Sealed for Owned {}
impl sealed::Sealed for View<'_> {}
impl sealed::Sealed for ViewMut<'_> {}

impl Repr for Owned {
    type Data<T: Element> = OwnedRepr<T>;
    type Attrs = Attrs;
    type Coord = Self;
}

impl<'a> Repr for View<'a> {
    type Data<T: Element> = ViewRepr<&'a T>;
    type Attrs = &'a Attrs;
    type Coord = Self;
}

impl<'a> Repr for ViewMut<'a> {
    type Data<T: Element> = ViewRepr<&'a mut T>;
    type Attrs = &'a Attrs;
    type Coord = View<'a>;
}

impl ReprMut for Owned {
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T> {
        a.view_mut()
    }
}

impl ReprMut for ViewMut<'_> {
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T> {
        a.view_mut()
    }
}
