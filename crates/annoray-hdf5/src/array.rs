//! [`Array`]: the values of a dataset or attribute, with their element type chosen at runtime.

use std::fmt;

use ndarray::{ArrayD, IxDyn};

/// The element type of an [`Array`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ValueType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    /// UTF-8 text (fixed- or variable-length strings).
    Str,
}

impl ValueType {
    /// Bytes per element in memory (a pointer's 8 for strings, as netCDF-C counts them).
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::Str => 8,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Str => "str",
        })
    }
}

/// An n-dimensional array of one of the element types HDF5 values are read as.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum Array {
    I8(ArrayD<i8>),
    U8(ArrayD<u8>),
    I16(ArrayD<i16>),
    U16(ArrayD<u16>),
    I32(ArrayD<i32>),
    U32(ArrayD<u32>),
    I64(ArrayD<i64>),
    U64(ArrayD<u64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Str(ArrayD<String>),
}

macro_rules! each {
    ($a:expr, $x:ident => $e:expr) => {
        match $a {
            Array::I8($x) => $e,
            Array::U8($x) => $e,
            Array::I16($x) => $e,
            Array::U16($x) => $e,
            Array::I32($x) => $e,
            Array::U32($x) => $e,
            Array::I64($x) => $e,
            Array::U64($x) => $e,
            Array::F32($x) => $e,
            Array::F64($x) => $e,
            Array::Str($x) => $e,
        }
    };
}

impl Array {
    /// The element type.
    #[must_use]
    pub fn dtype(&self) -> ValueType {
        match self {
            Self::I8(_) => ValueType::I8,
            Self::U8(_) => ValueType::U8,
            Self::I16(_) => ValueType::I16,
            Self::U16(_) => ValueType::U16,
            Self::I32(_) => ValueType::I32,
            Self::U32(_) => ValueType::U32,
            Self::I64(_) => ValueType::I64,
            Self::U64(_) => ValueType::U64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Str(_) => ValueType::Str,
        }
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        each!(self, a => a.shape())
    }

    /// The number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        each!(self, a => a.ndim())
    }

    /// The number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        each!(self, a => a.len())
    }

    /// Whether there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An array of `dtype` with no elements (`None` unless some length in `shape` is 0).
    #[must_use]
    pub fn empty(dtype: ValueType, shape: &[usize]) -> Option<Self> {
        if shape.iter().product::<usize>() != 0 {
            return None;
        }
        let sh = IxDyn(shape);
        macro_rules! e {
            ($($v:ident),*) => {
                match dtype {
                    $(ValueType::$v => Self::$v(ArrayD::from_shape_vec(sh, Vec::new()).ok()?),)*
                }
            };
        }
        Some(e!(I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Str))
    }
}

macro_rules! from_array {
    ($($v:ident: $t:ty),*) => {$(
        impl From<ArrayD<$t>> for Array {
            fn from(a: ArrayD<$t>) -> Self {
                Self::$v(a)
            }
        }
    )*};
}

from_array!(I8: i8, U8: u8, I16: i16, U16: u16, I32: i32, U32: u32, I64: i64, U64: u64,
    F32: f32, F64: f64, Str: String);
