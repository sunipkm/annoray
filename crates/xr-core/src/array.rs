//! [`TypedArray`]: an n-dimensional array whose element type is chosen at runtime.

use std::fmt;

use ndarray::{ArrayBase, Axis, IxDyn, Slice};

use crate::repr::{Owned, Repr, ReprMut, View, ViewMut};

/// An ndarray with dynamic dimensionality stored in representation `S`.
pub type Arr<S, T> = ArrayBase<<S as Repr>::Data<T>, IxDyn>;

/// The element type of a [`TypedArray`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Str,
    /// `chrono::NaiveDateTime` (real-world calendars).
    DateTime,
    /// [`CfDate`](xr_units::CfDate) (any CF calendar).
    CfDate,
    /// `chrono::TimeDelta`.
    TimeDelta,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::DateTime => "datetime",
            Self::CfDate => "cftime",
            Self::TimeDelta => "timedelta",
        })
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type that can be stored in a [`TypedArray`]. Sealed.
pub trait Element:
    sealed::Sealed + Clone + PartialEq + fmt::Debug + fmt::Display + Send + Sync + 'static
{
    /// The runtime tag for this type.
    const DTYPE: DType;

    /// Wrap a concretely typed array.
    fn wrap<S: Repr>(a: Arr<S, Self>) -> TypedArray<S>;

    /// Borrow the concretely typed array, if `a` holds this element type.
    fn peek<S: Repr>(a: &TypedArray<S>) -> Option<&Arr<S, Self>>;

    /// Mutably borrow the concretely typed array, if `a` holds this element type.
    fn peek_mut<S: Repr>(a: &mut TypedArray<S>) -> Option<&mut Arr<S, Self>>;
}

/// An n-dimensional array with a runtime element type, in storage representation `S`.
#[non_exhaustive]
#[allow(missing_docs)]
pub enum TypedArray<S: Repr> {
    F32(Arr<S, f32>),
    F64(Arr<S, f64>),
    I8(Arr<S, i8>),
    I16(Arr<S, i16>),
    I32(Arr<S, i32>),
    I64(Arr<S, i64>),
    U8(Arr<S, u8>),
    U16(Arr<S, u16>),
    U32(Arr<S, u32>),
    U64(Arr<S, u64>),
    Bool(Arr<S, bool>),
    Str(Arr<S, String>),
    DateTime(Arr<S, chrono::NaiveDateTime>),
    CfDate(Arr<S, xr_units::CfDate>),
    TimeDelta(Arr<S, chrono::TimeDelta>),
}

/// Run `$body` with `$a` bound to the inner array, whatever its element type.
macro_rules! typed_dispatch {
    ($value:expr, $a:ident => $body:expr) => {
        match $value {
            $crate::TypedArray::F32($a) => $body,
            $crate::TypedArray::F64($a) => $body,
            $crate::TypedArray::I8($a) => $body,
            $crate::TypedArray::I16($a) => $body,
            $crate::TypedArray::I32($a) => $body,
            $crate::TypedArray::I64($a) => $body,
            $crate::TypedArray::U8($a) => $body,
            $crate::TypedArray::U16($a) => $body,
            $crate::TypedArray::U32($a) => $body,
            $crate::TypedArray::U64($a) => $body,
            $crate::TypedArray::Bool($a) => $body,
            $crate::TypedArray::Str($a) => $body,
            $crate::TypedArray::DateTime($a) => $body,
            $crate::TypedArray::CfDate($a) => $body,
            $crate::TypedArray::TimeDelta($a) => $body,
        }
    };
}

/// Map the inner array to a new array of the same element type (possibly another repr).
macro_rules! typed_map {
    ($value:expr, $a:ident => $body:expr) => {
        match $value {
            $crate::TypedArray::F32($a) => $crate::TypedArray::F32($body),
            $crate::TypedArray::F64($a) => $crate::TypedArray::F64($body),
            $crate::TypedArray::I8($a) => $crate::TypedArray::I8($body),
            $crate::TypedArray::I16($a) => $crate::TypedArray::I16($body),
            $crate::TypedArray::I32($a) => $crate::TypedArray::I32($body),
            $crate::TypedArray::I64($a) => $crate::TypedArray::I64($body),
            $crate::TypedArray::U8($a) => $crate::TypedArray::U8($body),
            $crate::TypedArray::U16($a) => $crate::TypedArray::U16($body),
            $crate::TypedArray::U32($a) => $crate::TypedArray::U32($body),
            $crate::TypedArray::U64($a) => $crate::TypedArray::U64($body),
            $crate::TypedArray::Bool($a) => $crate::TypedArray::Bool($body),
            $crate::TypedArray::Str($a) => $crate::TypedArray::Str($body),
            $crate::TypedArray::DateTime($a) => $crate::TypedArray::DateTime($body),
            $crate::TypedArray::CfDate($a) => $crate::TypedArray::CfDate($body),
            $crate::TypedArray::TimeDelta($a) => $crate::TypedArray::TimeDelta($body),
        }
    };
}

pub(crate) use typed_dispatch;

/// Carries an element type chosen by `typed_dispatch!` into generic code:
/// `typed_dispatch!(a, x => f(..., marker(x)))` calls `f::<T>` for `a`'s element type.
pub(crate) struct Marker<T>(std::marker::PhantomData<T>);

pub(crate) fn marker<S: ndarray::Data, D>(_: &ndarray::ArrayBase<S, D>) -> Marker<S::Elem> {
    Marker(std::marker::PhantomData)
}

macro_rules! impl_element {
    ($($var:ident $ty:ty),* $(,)?) => {$(
        impl sealed::Sealed for $ty {}

        impl Element for $ty {
            const DTYPE: DType = DType::$var;

            fn wrap<S: Repr>(a: Arr<S, Self>) -> TypedArray<S> {
                TypedArray::$var(a)
            }

            fn peek<S: Repr>(a: &TypedArray<S>) -> Option<&Arr<S, Self>> {
                match a {
                    TypedArray::$var(x) => Some(x),
                    _ => None,
                }
            }

            fn peek_mut<S: Repr>(a: &mut TypedArray<S>) -> Option<&mut Arr<S, Self>> {
                match a {
                    TypedArray::$var(x) => Some(x),
                    _ => None,
                }
            }
        }
    )*};
}

impl_element!(
    F32 f32, F64 f64, I8 i8, I16 i16, I32 i32, I64 i64,
    U8 u8, U16 u16, U32 u32, U64 u64, Bool bool, Str String,
    DateTime chrono::NaiveDateTime, CfDate xr_units::CfDate, TimeDelta chrono::TimeDelta,
);

impl<S: Repr> TypedArray<S> {
    /// The runtime element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::Bool(_) => DType::Bool,
            Self::Str(_) => DType::Str,
            Self::DateTime(_) => DType::DateTime,
            Self::CfDate(_) => DType::CfDate,
            Self::TimeDelta(_) => DType::TimeDelta,
        }
    }

    /// Length of each axis.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        typed_dispatch!(self, a => a.shape())
    }

    /// Number of axes.
    #[must_use]
    pub fn ndim(&self) -> usize {
        typed_dispatch!(self, a => a.ndim())
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        typed_dispatch!(self, a => a.len())
    }

    /// Whether the array has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        typed_dispatch!(self, a => a.is_empty())
    }

    /// Borrow as a shared view.
    #[must_use]
    pub fn view(&self) -> TypedArray<View<'_>> {
        typed_map!(self, a => a.view())
    }

    /// Copy into an owned array.
    #[must_use]
    pub fn to_owned(&self) -> TypedArray<Owned> {
        typed_map!(self, a => a.to_owned())
    }

    /// Collapse `axis` at `index`, removing the axis.
    ///
    /// Callers must have validated `axis < ndim` and `index < shape[axis]`; ndarray panics otherwise.
    pub(crate) fn index_axis_move(self, axis: usize, index: usize) -> Self {
        typed_map!(self, a => a.index_axis_move(Axis(axis), index))
    }

    /// Copy the given positions along `axis` (fancy indexing).
    ///
    /// Callers must have validated every index against the axis length; ndarray panics otherwise.
    pub(crate) fn select_axis(&self, axis: usize, indices: &[usize]) -> TypedArray<Owned> {
        typed_map!(self, a => a.select(Axis(axis), indices))
    }

    /// Slice `axis` in place.
    ///
    /// Callers must have validated `slice` against the axis length (see `SliceIndices`).
    pub(crate) fn slice_axis_inplace(&mut self, axis: usize, slice: Slice) {
        typed_dispatch!(self, a => a.slice_axis_inplace(Axis(axis), slice));
    }
}

impl<S: ReprMut> TypedArray<S> {
    /// Borrow as a mutable view.
    #[must_use]
    pub fn view_mut(&mut self) -> TypedArray<ViewMut<'_>> {
        typed_map!(self, a => S::view_mut(a))
    }
}

impl<S: Repr> fmt::Debug for TypedArray<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dtype())?;
        typed_dispatch!(self, a => fmt::Debug::fmt(a, f))
    }
}

impl Clone for TypedArray<Owned> {
    fn clone(&self) -> Self {
        typed_map!(self, a => a.clone())
    }
}

impl Clone for TypedArray<View<'_>> {
    fn clone(&self) -> Self {
        typed_map!(self, a => a.clone())
    }
}

impl<T: Element> From<ndarray::ArrayD<T>> for TypedArray<Owned> {
    fn from(a: ndarray::ArrayD<T>) -> Self {
        T::wrap(a)
    }
}

impl<'a, T: Element> From<ndarray::ArrayViewD<'a, T>> for TypedArray<View<'a>> {
    fn from(a: ndarray::ArrayViewD<'a, T>) -> Self {
        T::wrap(a)
    }
}

impl<'a, T: Element> From<ndarray::ArrayViewMutD<'a, T>> for TypedArray<ViewMut<'a>> {
    fn from(a: ndarray::ArrayViewMutD<'a, T>) -> Self {
        T::wrap(a)
    }
}
