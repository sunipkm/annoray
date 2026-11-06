//! Storage representations: the `S` in `VariableBase<S>` / `DatasetBase<S>`.
//!
//! Like ndarray's `OwnedRepr` / `ViewRepr`, a [`Repr`] decides whether a container owns its
//! buffers, borrows them, or borrows them mutably. It does this for every element type at once
//! through the generic associated type [`Repr::Data`].

use std::borrow::Borrow;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use ndarray::{ArrayBase, ArrayViewMutD, IxDyn, OwnedRepr, ViewRepr};

use crate::{Attrs, DType, Element, TypedArray};

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
    /// How a variable holds its loaded values: shared and copied on write when owned, the
    /// array itself otherwise.
    #[doc(hidden)]
    type Held: Debug;
    #[doc(hidden)]
    fn held_shape(h: &Self::Held) -> &[usize];
    #[doc(hidden)]
    fn held_dtype(h: &Self::Held) -> DType;
    /// The values, as a view.
    #[doc(hidden)]
    fn held_typed(h: &Self::Held) -> TypedArray<View<'_>>;
    /// The values, as a view that remembers the buffer it shows all of.
    #[doc(hidden)]
    fn held_view(h: &Self::Held) -> HeldView<'_>;
    /// The values, as owned data: a new reference to the buffer if there is one to share.
    #[doc(hidden)]
    fn held_to_buffer(h: &Self::Held) -> SharedBuffer;
    /// The array itself, to reshape in place (owned data still shared is copied first).
    #[doc(hidden)]
    fn held_into_typed(h: Self::Held) -> TypedArray<Self>;
    #[doc(hidden)]
    fn held_from_typed(a: TypedArray<Self>) -> Self::Held;
}

/// A representation that allows mutating element data.
pub trait ReprMut: Repr {
    /// Mutably view an array stored in this representation.
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T>;
    /// The array to write: owned data still shared with others is copied first.
    #[doc(hidden)]
    fn held_mut(h: &mut Self::Held) -> &mut TypedArray<Self>;
}

/// A read-only array in memory that annoray does not own (a buffer of another library, kept
/// alive by whoever implements this).
///
/// Variables made with [`Variable::from_external`](crate::Variable::from_external) read it in
/// place. Writing to such a variable first copies the values into annoray's own memory.
///
/// The trait is safe to implement; an implementation that borrows foreign memory has to
/// uphold, itself, that the memory stays valid and unwritten for as long as the implementor
/// lives.
pub trait ExternalBuffer: Send + Sync + 'static {
    /// The values. Every call shows the same memory.
    fn typed(&self) -> TypedArray<View<'_>>;
}

#[derive(Clone)]
enum Stored {
    Mem(Arc<TypedArray<Owned>>),
    External {
        buffer: Arc<dyn ExternalBuffer>,
        shape: Arc<[usize]>,
        dtype: DType,
        /// The values once written to: `buffer` is never written, and is not read after this.
        written: Option<Arc<TypedArray<Owned>>>,
    },
}

/// Loaded values of an owned variable, reference counted: cloning a variable shares them, and
/// writing through it copies them first if anything else still holds them (copy on write).
#[derive(Clone)]
pub struct SharedBuffer(Stored);

impl SharedBuffer {
    pub(crate) fn new(a: TypedArray<Owned>) -> Self {
        Self(Stored::Mem(Arc::new(a)))
    }

    pub(crate) fn external(buffer: Arc<dyn ExternalBuffer>) -> Self {
        let (shape, dtype) = {
            let t = buffer.typed();
            (Arc::from(t.shape()), t.dtype())
        };
        Self(Stored::External {
            buffer,
            shape,
            dtype,
            written: None,
        })
    }

    /// The values, as a view.
    pub(crate) fn typed(&self) -> TypedArray<View<'_>> {
        match &self.0 {
            Stored::Mem(a)
            | Stored::External {
                written: Some(a), ..
            } => a.view(),
            Stored::External { buffer, .. } => buffer.typed(),
        }
    }

    /// The array, moved out if nothing else holds it and copied otherwise.
    pub(crate) fn into_typed(self) -> TypedArray<Owned> {
        match self.0 {
            Stored::Mem(a)
            | Stored::External {
                written: Some(a), ..
            } => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
            Stored::External { buffer, .. } => buffer.typed().to_owned(),
        }
    }

    /// The array to write: annoray's own, and not shared with anything else.
    pub(crate) fn make_mut(&mut self) -> &mut TypedArray<Owned> {
        Arc::make_mut(match &mut self.0 {
            Stored::Mem(a) => a,
            Stored::External {
                buffer, written, ..
            } => written.get_or_insert_with(|| Arc::new(buffer.typed().to_owned())),
        })
    }

    /// An independent copy in annoray's own memory.
    pub(crate) fn deep_copy(&self) -> Self {
        Self::new(self.typed().to_owned())
    }

    /// An owner of the memory [`typed`](Self::typed) shows.
    pub(crate) fn keep_alive(&self) -> KeepAlive {
        match &self.0 {
            Stored::Mem(a)
            | Stored::External {
                written: Some(a), ..
            } => a.clone(),
            Stored::External { buffer, .. } => Arc::new(buffer.clone()),
        }
    }
}

/// Keeps a variable's memory valid and unmoved for as long as it lives, whatever happens to
/// the variable. See [`Variable::keep_alive`](crate::Variable::keep_alive).
pub type KeepAlive = Arc<dyn std::any::Any + Send + Sync>;

impl Debug for SharedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.typed(), f)
    }
}

/// Loaded values of a shared view: the view, and the buffer it shows all of, if it does. An
/// unsliced view made owned ([`to_owned`](crate::VariableBase::to_owned)) shares that buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HeldView<'a> {
    typed: TypedArray<View<'a>>,
    whole: Option<SharedBuffer>,
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
    type Held = SharedBuffer;
    fn held_shape(h: &Self::Held) -> &[usize] {
        match &h.0 {
            Stored::Mem(a) => a.shape(),
            Stored::External { shape, .. } => shape,
        }
    }
    fn held_dtype(h: &Self::Held) -> DType {
        match &h.0 {
            Stored::Mem(a) => a.dtype(),
            Stored::External { dtype, .. } => *dtype,
        }
    }
    fn held_typed(h: &Self::Held) -> TypedArray<View<'_>> {
        h.typed()
    }
    fn held_view(h: &Self::Held) -> HeldView<'_> {
        HeldView {
            typed: h.typed(),
            whole: Some(h.clone()),
        }
    }
    fn held_to_buffer(h: &Self::Held) -> SharedBuffer {
        h.clone()
    }
    fn held_into_typed(h: Self::Held) -> TypedArray<Self> {
        h.into_typed()
    }
    fn held_from_typed(a: TypedArray<Self>) -> Self::Held {
        SharedBuffer::new(a)
    }
}

impl<'a> Repr for View<'a> {
    type Data<T: Element> = ViewRepr<&'a T>;
    type Attrs = &'a Attrs;
    type Coord = Self;
    type Held = HeldView<'a>;
    fn held_shape(h: &Self::Held) -> &[usize] {
        h.typed.shape()
    }
    fn held_dtype(h: &Self::Held) -> DType {
        h.typed.dtype()
    }
    fn held_typed(h: &Self::Held) -> TypedArray<View<'_>> {
        h.typed.view()
    }
    fn held_view(h: &Self::Held) -> HeldView<'_> {
        HeldView {
            typed: h.typed.view(),
            whole: h.whole.clone(),
        }
    }
    fn held_to_buffer(h: &Self::Held) -> SharedBuffer {
        match &h.whole {
            Some(b) => b.clone(),
            None => SharedBuffer::new(h.typed.to_owned()),
        }
    }
    fn held_into_typed(h: Self::Held) -> TypedArray<Self> {
        h.typed
    }
    fn held_from_typed(a: TypedArray<Self>) -> Self::Held {
        HeldView {
            typed: a,
            whole: None,
        }
    }
}

impl<'a> Repr for ViewMut<'a> {
    type Data<T: Element> = ViewRepr<&'a mut T>;
    type Attrs = &'a Attrs;
    type Coord = View<'a>;
    type Held = TypedArray<Self>;
    fn held_shape(h: &Self::Held) -> &[usize] {
        h.shape()
    }
    fn held_dtype(h: &Self::Held) -> DType {
        h.dtype()
    }
    fn held_typed(h: &Self::Held) -> TypedArray<View<'_>> {
        h.view()
    }
    fn held_view(h: &Self::Held) -> HeldView<'_> {
        HeldView {
            typed: h.view(),
            whole: None,
        }
    }
    fn held_to_buffer(h: &Self::Held) -> SharedBuffer {
        SharedBuffer::new(h.to_owned())
    }
    fn held_into_typed(h: Self::Held) -> TypedArray<Self> {
        h
    }
    fn held_from_typed(a: TypedArray<Self>) -> Self::Held {
        a
    }
}

impl ReprMut for Owned {
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T> {
        a.view_mut()
    }
    fn held_mut(h: &mut Self::Held) -> &mut TypedArray<Self> {
        h.make_mut()
    }
}

impl ReprMut for ViewMut<'_> {
    fn view_mut<T: Element>(a: &mut ArrayBase<Self::Data<T>, IxDyn>) -> ArrayViewMutD<'_, T> {
        a.view_mut()
    }
    fn held_mut(h: &mut Self::Held) -> &mut TypedArray<Self> {
        h
    }
}
