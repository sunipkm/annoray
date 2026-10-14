//! netCDF-3 external types and big-endian (de)serialisation of values.

use ndarray::{ArrayD, IxDyn};
use xr_core::{AttrValue, DType, Owned, TypedArray};
use xr_io::IoError;

use crate::{Version, bad};

pub(crate) const NC_DIMENSION: i32 = 10;
pub(crate) const NC_VARIABLE: i32 = 11;
pub(crate) const NC_ATTRIBUTE: i32 = 12;

/// An external data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
    UByte,
    UShort,
    UInt,
    Int64,
    UInt64,
}

impl NcType {
    /// The element type values are read as (characters are bytes).
    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::Byte => DType::I8,
            Self::Char | Self::UByte => DType::U8,
            Self::Short => DType::I16,
            Self::UShort => DType::U16,
            Self::Int => DType::I32,
            Self::UInt => DType::U32,
            Self::Int64 => DType::I64,
            Self::UInt64 => DType::U64,
            Self::Float => DType::F32,
            Self::Double => DType::F64,
        }
    }

    pub(crate) fn from_code(code: i32, v: Version) -> Option<Self> {
        let t = match code {
            1 => Self::Byte,
            2 => Self::Char,
            3 => Self::Short,
            4 => Self::Int,
            5 => Self::Float,
            6 => Self::Double,
            7 => Self::UByte,
            8 => Self::UShort,
            9 => Self::UInt,
            10 => Self::Int64,
            11 => Self::UInt64,
            _ => return None,
        };
        (v == Version::Cdf5 || !t.cdf5_only()).then_some(t)
    }

    pub(crate) const fn code(self) -> i32 {
        match self {
            Self::Byte => 1,
            Self::Char => 2,
            Self::Short => 3,
            Self::Int => 4,
            Self::Float => 5,
            Self::Double => 6,
            Self::UByte => 7,
            Self::UShort => 8,
            Self::UInt => 9,
            Self::Int64 => 10,
            Self::UInt64 => 11,
        }
    }

    pub(crate) const fn size(self) -> usize {
        match self {
            Self::Byte | Self::Char | Self::UByte => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double | Self::Int64 | Self::UInt64 => 8,
        }
    }

    const fn cdf5_only(self) -> bool {
        matches!(
            self,
            Self::UByte | Self::UShort | Self::UInt | Self::Int64 | Self::UInt64
        )
    }

    /// The external type for an in-memory element type, if the version has one.
    pub(crate) fn of(d: DType, is_char: bool, v: Version) -> Option<Self> {
        let t = match (d, is_char) {
            (DType::U8, true) => Self::Char,
            (DType::I8, _) => Self::Byte,
            (DType::I16, _) => Self::Short,
            (DType::I32, _) => Self::Int,
            (DType::F32, _) => Self::Float,
            (DType::F64, _) => Self::Double,
            (DType::U8, false) => Self::UByte,
            (DType::U16, _) => Self::UShort,
            (DType::U32, _) => Self::UInt,
            (DType::I64, _) => Self::Int64,
            (DType::U64, _) => Self::UInt64,
            _ => return None,
        };
        (v == Version::Cdf5 || !t.cdf5_only()).then_some(t)
    }
}

macro_rules! from_be {
    ($bytes:expr, $t:ty, $n:literal) => {
        $bytes
            .chunks_exact($n)
            .map(|c| <$t>::from_be_bytes(c.try_into().unwrap_or([0; $n])))
            .collect::<Vec<$t>>()
    };
}

/// Decode `bytes` (exactly `shape.product() * ty.size()` long) into an array.
pub(crate) fn decode_array(
    ty: NcType,
    bytes: &[u8],
    shape: &[usize],
) -> Result<TypedArray<Owned>, IoError> {
    let sh = IxDyn(shape);
    let err = |e: ndarray::ShapeError| bad(format!("data size does not match its shape: {e}"));
    Ok(match ty {
        NcType::Byte => {
            TypedArray::I8(ArrayD::from_shape_vec(sh, from_be!(bytes, i8, 1)).map_err(err)?)
        }
        NcType::Char | NcType::UByte => {
            TypedArray::U8(ArrayD::from_shape_vec(sh, bytes.to_vec()).map_err(err)?)
        }
        NcType::Short => {
            TypedArray::I16(ArrayD::from_shape_vec(sh, from_be!(bytes, i16, 2)).map_err(err)?)
        }
        NcType::UShort => {
            TypedArray::U16(ArrayD::from_shape_vec(sh, from_be!(bytes, u16, 2)).map_err(err)?)
        }
        NcType::Int => {
            TypedArray::I32(ArrayD::from_shape_vec(sh, from_be!(bytes, i32, 4)).map_err(err)?)
        }
        NcType::UInt => {
            TypedArray::U32(ArrayD::from_shape_vec(sh, from_be!(bytes, u32, 4)).map_err(err)?)
        }
        NcType::Float => {
            TypedArray::F32(ArrayD::from_shape_vec(sh, from_be!(bytes, f32, 4)).map_err(err)?)
        }
        NcType::Double => {
            TypedArray::F64(ArrayD::from_shape_vec(sh, from_be!(bytes, f64, 8)).map_err(err)?)
        }
        NcType::Int64 => {
            TypedArray::I64(ArrayD::from_shape_vec(sh, from_be!(bytes, i64, 8)).map_err(err)?)
        }
        NcType::UInt64 => {
            TypedArray::U64(ArrayD::from_shape_vec(sh, from_be!(bytes, u64, 8)).map_err(err)?)
        }
    })
}

/// Decode an attribute's values.
pub(crate) fn decode_attr(ty: NcType, bytes: &[u8]) -> AttrValue {
    match ty {
        NcType::Char => {
            let end = bytes
                .iter()
                .rposition(|b| *b != 0)
                .map_or(0, |p| p.saturating_add(1));
            AttrValue::Str(
                String::from_utf8_lossy(bytes.get(..end).unwrap_or_default()).into_owned(),
            )
        }
        NcType::Byte => AttrValue::I8(from_be!(bytes, i8, 1)),
        NcType::UByte => AttrValue::U8(bytes.to_vec()),
        NcType::Short => AttrValue::I16(from_be!(bytes, i16, 2)),
        NcType::UShort => AttrValue::U16(from_be!(bytes, u16, 2)),
        NcType::Int => AttrValue::I32(from_be!(bytes, i32, 4)),
        NcType::UInt => AttrValue::U32(from_be!(bytes, u32, 4)),
        NcType::Float => AttrValue::F32(from_be!(bytes, f32, 4)),
        NcType::Double => AttrValue::F64(from_be!(bytes, f64, 8)),
        NcType::Int64 => AttrValue::I64(from_be!(bytes, i64, 8)),
        NcType::UInt64 => AttrValue::U64(from_be!(bytes, u64, 8)),
    }
}

/// Big-endian bytes of an array (logical order).
pub(crate) fn encode_array(a: &TypedArray<Owned>) -> Option<Vec<u8>> {
    macro_rules! be {
        ($x:expr) => {
            $x.iter().flat_map(|v| v.to_be_bytes()).collect()
        };
    }
    Some(match a {
        TypedArray::I8(x) => be!(x),
        TypedArray::U8(x) => x.iter().copied().collect(),
        TypedArray::I16(x) => be!(x),
        TypedArray::U16(x) => be!(x),
        TypedArray::I32(x) => be!(x),
        TypedArray::U32(x) => be!(x),
        TypedArray::I64(x) => be!(x),
        TypedArray::U64(x) => be!(x),
        TypedArray::F32(x) => be!(x),
        TypedArray::F64(x) => be!(x),
        _ => return None,
    })
}

/// An attribute's external type, element count and big-endian bytes.
pub(crate) fn encode_attr(a: &AttrValue, v: Version) -> Option<(NcType, usize, Vec<u8>)> {
    macro_rules! be {
        ($t:expr, $x:expr) => {
            (
                $t,
                $x.len(),
                $x.iter().flat_map(|v| v.to_be_bytes()).collect(),
            )
        };
    }
    let out = match a {
        AttrValue::Str(s) => (NcType::Char, s.len(), s.as_bytes().to_vec()),
        AttrValue::I8(x) => be!(NcType::Byte, x),
        AttrValue::I16(x) => be!(NcType::Short, x),
        AttrValue::I32(x) => be!(NcType::Int, x),
        AttrValue::F32(x) => be!(NcType::Float, x),
        AttrValue::F64(x) => be!(NcType::Double, x),
        AttrValue::U8(x) => (NcType::UByte, x.len(), x.clone()),
        AttrValue::U16(x) => be!(NcType::UShort, x),
        AttrValue::U32(x) => be!(NcType::UInt, x),
        AttrValue::I64(x) => be!(NcType::Int64, x),
        AttrValue::U64(x) => be!(NcType::UInt64, x),
        _ => return None,
    };
    (v == Version::Cdf5 || !out.0.cdf5_only()).then_some(out)
}

/// One element of the fill value netCDF-C uses for padding: the variable's `_FillValue` if it
/// has the variable's type, else the type's default fill.
pub(crate) fn fill_element(ty: NcType, fill_attr: Option<&AttrValue>) -> Vec<u8> {
    let own = fill_attr.and_then(|a| match a {
        AttrValue::I8(v) if ty == NcType::Byte => v.first().map(|x| x.to_be_bytes().to_vec()),
        AttrValue::U8(v) if ty == NcType::UByte => v.first().map(|x| vec![*x]),
        AttrValue::I16(v) if ty == NcType::Short => v.first().map(|x| x.to_be_bytes().to_vec()),
        AttrValue::U16(v) if ty == NcType::UShort => v.first().map(|x| x.to_be_bytes().to_vec()),
        _ => None,
    });
    own.unwrap_or_else(|| match ty {
        NcType::Byte => (-127i8).to_be_bytes().to_vec(),
        NcType::Char => vec![0],
        NcType::UByte => vec![255],
        NcType::Short => (-32767i16).to_be_bytes().to_vec(),
        NcType::UShort => 65535u16.to_be_bytes().to_vec(),
        // Wider types are always 4-byte aligned and never padded.
        _ => vec![0],
    })
}
