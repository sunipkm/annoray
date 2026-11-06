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
    /// The values after `steps`, without copying them.
    #[doc(hidden)]
    fn held_select(h: Self::Held, steps: &[Step]) -> Self::Held;
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

/// What a selection does to an array's axes. Callers have validated axes and positions.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum Step {
    /// Slice an axis, keeping it.
    Slice(usize, ndarray::Slice),
    /// Take one position of an axis, removing it.
    At(usize, usize),
    /// Reorder the axes.
    Permute(Vec<usize>),
}

fn apply_steps<S: Repr>(mut t: TypedArray<S>, steps: &[Step]) -> TypedArray<S> {
    for step in steps {
        t = match step {
            Step::Slice(axis, slice) => {
                t.slice_axis_inplace(*axis, *slice);
                t
            }
            Step::At(axis, index) => t.index_axis_move(*axis, *index),
            Step::Permute(order) => t.permuted(order),
        };
    }
    t
}

#[derive(Clone)]
enum Base {
    Mem(Arc<TypedArray<Owned>>),
    External {
        buffer: Arc<dyn ExternalBuffer>,
        shape: Arc<[usize]>,
        dtype: DType,
        /// The values once written to: `buffer` is never written, and is not read after this.
        written: Option<Arc<TypedArray<Owned>>>,
    },
}

/// A selection of a buffer: the steps from the whole, and the shape they give.
struct Window {
    steps: Vec<Step>,
    shape: Box<[usize]>,
}

/// Loaded values of an owned variable, reference counted: cloning a variable or selecting from
/// it shares them (a selection keeps the whole buffer alive), and writing through it copies them
/// first if anything else still holds them (copy on write).
#[derive(Clone)]
pub struct SharedBuffer {
    base: Base,
    window: Option<Arc<Window>>,
}

impl SharedBuffer {
    pub(crate) fn new(a: TypedArray<Owned>) -> Self {
        Self {
            base: Base::Mem(Arc::new(a)),
            window: None,
        }
    }

    pub(crate) fn external(buffer: Arc<dyn ExternalBuffer>) -> Self {
        let (shape, dtype) = {
            let t = buffer.typed();
            (Arc::from(t.shape()), t.dtype())
        };
        Self {
            base: Base::External {
                buffer,
                shape,
                dtype,
                written: None,
            },
            window: None,
        }
    }

    /// The whole buffer, as a view.
    fn base_view(&self) -> TypedArray<View<'_>> {
        match &self.base {
            Base::Mem(a)
            | Base::External {
                written: Some(a), ..
            } => a.view(),
            Base::External { buffer, .. } => buffer.typed(),
        }
    }

    /// The values, as a view.
    pub(crate) fn typed(&self) -> TypedArray<View<'_>> {
        match &self.window {
            Some(w) => apply_steps(self.base_view(), &w.steps),
            None => self.base_view(),
        }
    }

    fn shape(&self) -> &[usize] {
        match (&self.window, &self.base) {
            (Some(w), _) => &w.shape,
            (None, Base::Mem(a)) => a.shape(),
            (None, Base::External { shape, .. }) => shape,
        }
    }

    fn dtype(&self) -> DType {
        match &self.base {
            Base::Mem(a) => a.dtype(),
            Base::External { dtype, .. } => *dtype,
        }
    }

    /// The same memory after `steps`.
    pub(crate) fn select(&self, steps: &[Step]) -> Self {
        let mut all = self
            .window
            .as_ref()
            .map_or_else(Vec::new, |w| w.steps.clone());
        all.extend_from_slice(steps);
        let shape = apply_steps(self.base_view(), &all).shape().into();
        Self {
            base: self.base.clone(),
            window: Some(Arc::new(Window { steps: all, shape })),
        }
    }

    /// The array, moved out if nothing else holds all of it and copied otherwise.
    pub(crate) fn into_typed(self) -> TypedArray<Owned> {
        if self.window.is_some() {
            return self.typed().to_owned();
        }
        match self.base {
            Base::Mem(a)
            | Base::External {
                written: Some(a), ..
            } => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
            Base::External { buffer, .. } => buffer.typed().to_owned(),
        }
    }

    /// The array to write: annoray's own, holding these values alone and shared with nothing.
    pub(crate) fn make_mut(&mut self) -> &mut TypedArray<Owned> {
        if self.window.is_some() {
            *self = Self::new(self.typed().to_owned());
        }
        Arc::make_mut(match &mut self.base {
            Base::Mem(a) => a,
            Base::External {
                buffer, written, ..
            } => written.get_or_insert_with(|| Arc::new(buffer.typed().to_owned())),
        })
    }

    /// An independent copy in annoray's own memory, of these values alone.
    pub(crate) fn deep_copy(&self) -> Self {
        Self::new(self.typed().to_owned())
    }

    /// An owner of the memory [`typed`](Self::typed) shows.
    pub(crate) fn keep_alive(&self) -> KeepAlive {
        match &self.base {
            Base::Mem(a)
            | Base::External {
                written: Some(a), ..
            } => a.clone(),
            Base::External { buffer, .. } => Arc::new(buffer.clone()),
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

/// Loaded values of a shared view: the view, and the buffer it shows, if it knows it. A view
/// made owned ([`to_owned`](crate::VariableBase::to_owned)) then shares that buffer.
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
        h.shape()
    }
    fn held_dtype(h: &Self::Held) -> DType {
        h.dtype()
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
    fn held_select(h: Self::Held, steps: &[Step]) -> Self::Held {
        h.select(steps)
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
    fn held_select(h: Self::Held, steps: &[Step]) -> Self::Held {
        HeldView {
            whole: h.whole.map(|w| w.select(steps)),
            typed: apply_steps(h.typed, steps),
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
    fn held_select(h: Self::Held, steps: &[Step]) -> Self::Held {
        apply_steps(h, steps)
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
