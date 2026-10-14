//! Variable and dataset attributes.

use indexmap::IndexMap;

use crate::Name;

/// Ordered attribute map, as in netCDF.
pub type Attrs = IndexMap<Name, AttrValue>;

/// An attribute value. Numeric attributes are always 1-D, as in netCDF.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum AttrValue {
    Str(String),
    /// A list of strings (netCDF-4 `NC_STRING` attributes).
    Strs(Vec<String>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl From<&str> for AttrValue {
    fn from(s: &str) -> Self {
        Self::Str(s.to_owned())
    }
}

impl From<String> for AttrValue {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

macro_rules! numeric_attr {
    ($($var:ident $ty:ty),* $(,)?) => {$(
        impl From<$ty> for AttrValue {
            fn from(v: $ty) -> Self {
                Self::$var(vec![v])
            }
        }
        impl From<Vec<$ty>> for AttrValue {
            fn from(v: Vec<$ty>) -> Self {
                Self::$var(v)
            }
        }
        impl From<&[$ty]> for AttrValue {
            fn from(v: &[$ty]) -> Self {
                Self::$var(v.to_vec())
            }
        }
    )*};
}

numeric_attr!(
    I8 i8, I16 i16, I32 i32, I64 i64, U8 u8, U16 u16, U32 u32, U64 u64, F32 f32, F64 f64,
);
