//! Element types ([`DType`]) and values ([`ArrayData`]).

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use serde_json::Value;

use crate::{Error, bad};

/// An element type as stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// One byte, 0 or 1.
    Bool,
    /// Signed integers.
    I8,
    #[allow(missing_docs)]
    I16,
    #[allow(missing_docs)]
    I32,
    #[allow(missing_docs)]
    I64,
    /// Unsigned integers.
    U8,
    #[allow(missing_docs)]
    U16,
    #[allow(missing_docs)]
    U32,
    #[allow(missing_docs)]
    U64,
    /// IEEE floats.
    F32,
    #[allow(missing_docs)]
    F64,
    /// IEEE complex numbers: two `f32` parts (v2 `<c8`, v3 `complex64`), real part first.
    C64,
    /// Two `f64` parts (v2 `<c16`, v3 `complex128`).
    C128,
    /// Fixed-width bytes (v2 `|S<n>`), read as text (trailing NULs dropped).
    FixedBytes(usize),
    /// Fixed-width UCS-4 text of `n` characters (v2 `<U<n>`, v3 `fixed_length_utf32`).
    FixedUnicode(usize),
    /// Variable-length UTF-8 text (v2 `|O` with the `vlen-utf8` filter, v3 `string`).
    String,
}

/// Byte order of multi-byte elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    /// Little-endian.
    Little,
    /// Big-endian.
    Big,
}

impl DType {
    /// Bytes per element when stored by the `bytes` codec (`None` for variable-length text).
    #[must_use]
    pub fn size(self) -> Option<usize> {
        Some(match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
            Self::FixedBytes(n) => n,
            Self::FixedUnicode(n) => n.checked_mul(4)?,
            Self::String => return None,
        })
    }

    /// Parse a v2 `dtype` (`"<f8"`, `"|b1"`, `"|S5"`, `"<U3"`, `"|O"`), with its byte order.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for other types (structured, datetimes).
    pub fn from_v2(s: &str) -> Result<(Self, Endian), Error> {
        let unsupported = || Error::Unsupported(format!("zarr v2 dtype {s:?}"));
        let mut chars = s.chars();
        let order = chars.next().ok_or_else(unsupported)?;
        let rest: String = chars.collect();
        let endian = if order == '>' {
            Endian::Big
        } else {
            Endian::Little
        };
        if !matches!(order, '<' | '>' | '|' | '=') {
            return Err(unsupported());
        }
        let kind = rest.chars().next().ok_or_else(unsupported)?;
        let width: usize = rest.get(1..).unwrap_or("").parse().unwrap_or(0);
        let t = match (kind, width) {
            ('b', 1) => Self::Bool,
            ('i', 1) => Self::I8,
            ('i', 2) => Self::I16,
            ('i', 4) => Self::I32,
            ('i', 8) => Self::I64,
            ('u', 1) => Self::U8,
            ('u', 2) => Self::U16,
            ('u', 4) => Self::U32,
            ('u', 8) => Self::U64,
            ('f', 4) => Self::F32,
            ('f', 8) => Self::F64,
            ('c', 8) => Self::C64,
            ('c', 16) => Self::C128,
            ('S', n) => Self::FixedBytes(n),
            ('U', n) => Self::FixedUnicode(n),
            ('O', _) => Self::String,
            _ => return Err(unsupported()),
        };
        Ok((t, endian))
    }

    /// The v2 `dtype` string (little-endian).
    #[must_use]
    pub fn to_v2(self) -> String {
        match self {
            Self::Bool => "|b1".into(),
            Self::I8 => "|i1".into(),
            Self::I16 => "<i2".into(),
            Self::I32 => "<i4".into(),
            Self::I64 => "<i8".into(),
            Self::U8 => "|u1".into(),
            Self::U16 => "<u2".into(),
            Self::U32 => "<u4".into(),
            Self::U64 => "<u8".into(),
            Self::F32 => "<f4".into(),
            Self::F64 => "<f8".into(),
            Self::C64 => "<c8".into(),
            Self::C128 => "<c16".into(),
            Self::FixedBytes(n) => format!("|S{n}"),
            Self::FixedUnicode(n) => format!("<U{n}"),
            Self::String => "|O".into(),
        }
    }

    /// Parse a v3 `data_type` (`"float64"`, `"string"`, or an object such as
    /// `{"name": "fixed_length_utf32", "configuration": {"length_bytes": 12}}`).
    ///
    /// # Errors
    /// [`Error::Unsupported`] for other types.
    pub fn from_v3(v: &Value) -> Result<Self, Error> {
        let unsupported = || Error::Unsupported(format!("zarr v3 data_type {v}"));
        let (name, conf) = match v {
            Value::String(s) => (s.as_str(), None),
            Value::Object(o) => (
                o.get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(unsupported)?,
                o.get("configuration"),
            ),
            _ => return Err(unsupported()),
        };
        Ok(match name {
            "bool" => Self::Bool,
            "int8" => Self::I8,
            "int16" => Self::I16,
            "int32" => Self::I32,
            "int64" => Self::I64,
            "uint8" => Self::U8,
            "uint16" => Self::U16,
            "uint32" => Self::U32,
            "uint64" => Self::U64,
            "float32" => Self::F32,
            "float64" => Self::F64,
            "complex64" => Self::C64,
            "complex128" => Self::C128,
            "string" | "variable_length_utf8" => Self::String,
            "fixed_length_utf32" => {
                let bytes = conf
                    .and_then(|c| c.get("length_bytes"))
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(unsupported)?;
                Self::FixedUnicode(bytes / 4)
            }
            "fixed_length_bytes" | "null_terminated_bytes" => {
                let bytes = conf
                    .and_then(|c| c.get("length_bytes"))
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(unsupported)?;
                Self::FixedBytes(bytes)
            }
            _ => return Err(unsupported()),
        })
    }

    /// The v3 `data_type`.
    #[must_use]
    pub fn to_v3(self) -> Value {
        let s = match self {
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::C64 => "complex64",
            Self::C128 => "complex128",
            Self::String => "string",
            Self::FixedUnicode(n) => {
                return serde_json::json!({
                    "name": "fixed_length_utf32",
                    "configuration": {"length_bytes": n.saturating_mul(4)}
                });
            }
            Self::FixedBytes(n) => {
                return serde_json::json!({
                    "name": "null_terminated_bytes",
                    "configuration": {"length_bytes": n}
                });
            }
        };
        Value::String(s.into())
    }
}

/// Values of an array or a region of one, with their element type chosen at runtime. Text of
/// every kind is `Str`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ArrayData {
    Bool(ArrayD<bool>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    C64(ArrayD<Complex<f32>>),
    C128(ArrayD<Complex<f64>>),
    Str(ArrayD<String>),
}

/// Run `$body` with `$a` bound to the inner array.
macro_rules! each {
    ($v:expr, $a:ident => $body:expr) => {
        match $v {
            ArrayData::Bool($a) => $body,
            ArrayData::I8($a) => $body,
            ArrayData::I16($a) => $body,
            ArrayData::I32($a) => $body,
            ArrayData::I64($a) => $body,
            ArrayData::U8($a) => $body,
            ArrayData::U16($a) => $body,
            ArrayData::U32($a) => $body,
            ArrayData::U64($a) => $body,
            ArrayData::F32($a) => $body,
            ArrayData::F64($a) => $body,
            ArrayData::C64($a) => $body,
            ArrayData::C128($a) => $body,
            ArrayData::Str($a) => $body,
        }
    };
}

/// Map the inner array to another of the same element type.
macro_rules! map_each {
    ($v:expr, $a:ident => $body:expr) => {
        match $v {
            ArrayData::Bool($a) => ArrayData::Bool($body),
            ArrayData::I8($a) => ArrayData::I8($body),
            ArrayData::I16($a) => ArrayData::I16($body),
            ArrayData::I32($a) => ArrayData::I32($body),
            ArrayData::I64($a) => ArrayData::I64($body),
            ArrayData::U8($a) => ArrayData::U8($body),
            ArrayData::U16($a) => ArrayData::U16($body),
            ArrayData::U32($a) => ArrayData::U32($body),
            ArrayData::U64($a) => ArrayData::U64($body),
            ArrayData::F32($a) => ArrayData::F32($body),
            ArrayData::F64($a) => ArrayData::F64($body),
            ArrayData::C64($a) => ArrayData::C64($body),
            ArrayData::C128($a) => ArrayData::C128($body),
            ArrayData::Str($a) => ArrayData::Str($body),
        }
    };
}
#[allow(unused_imports)]
pub(crate) use {each, map_each};

impl ArrayData {
    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        each!(self, a => a.shape())
    }

    /// The element type of the values in memory (text as [`DType::String`]).
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::C64(_) => DType::C64,
            Self::C128(_) => DType::C128,
            Self::Str(_) => DType::String,
        }
    }

    /// An array of `shape` filled with `fill` (which must be of a matching type).
    pub(crate) fn filled(dtype: DType, shape: &[usize], fill: &Fill) -> Result<Self, Error> {
        let sh = IxDyn(shape);
        Ok(match (dtype, fill) {
            (DType::Bool, Fill::Bool(v)) => Self::Bool(ArrayD::from_elem(sh, *v)),
            (DType::I8, Fill::Int(v)) => Self::I8(ArrayD::from_elem(sh, int(*v)?)),
            (DType::I16, Fill::Int(v)) => Self::I16(ArrayD::from_elem(sh, int(*v)?)),
            (DType::I32, Fill::Int(v)) => Self::I32(ArrayD::from_elem(sh, int(*v)?)),
            (DType::I64, Fill::Int(v)) => Self::I64(ArrayD::from_elem(sh, int(*v)?)),
            (DType::U8, Fill::Int(v)) => Self::U8(ArrayD::from_elem(sh, int(*v)?)),
            (DType::U16, Fill::Int(v)) => Self::U16(ArrayD::from_elem(sh, int(*v)?)),
            (DType::U32, Fill::Int(v)) => Self::U32(ArrayD::from_elem(sh, int(*v)?)),
            (DType::U64, Fill::Int(v)) => Self::U64(ArrayD::from_elem(sh, int(*v)?)),
            #[allow(clippy::cast_possible_truncation)]
            (DType::F32, Fill::Float(v)) => Self::F32(ArrayD::from_elem(sh, *v as f32)),
            (DType::F64, Fill::Float(v)) => Self::F64(ArrayD::from_elem(sh, *v)),
            #[allow(clippy::cast_possible_truncation)]
            (DType::C64, Fill::Complex(re, im)) => {
                Self::C64(ArrayD::from_elem(sh, Complex::new(*re as f32, *im as f32)))
            }
            (DType::C128, Fill::Complex(re, im)) => {
                Self::C128(ArrayD::from_elem(sh, Complex::new(*re, *im)))
            }
            (DType::FixedBytes(_) | DType::FixedUnicode(_) | DType::String, Fill::Str(v)) => {
                Self::Str(ArrayD::from_elem(sh, v.clone()))
            }
            (d, f) => return Err(bad(format!("fill value {f:?} for {d:?}"))),
        })
    }
}

fn int<T: TryFrom<i128>>(v: i128) -> Result<T, Error> {
    T::try_from(v).map_err(|_| bad(format!("fill value {v} out of range")))
}

/// A fill value, parsed for an element type.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Fill {
    /// For booleans.
    Bool(bool),
    /// For integers.
    Int(i128),
    /// For floats (NaN and infinities included).
    Float(f64),
    /// For complex numbers: the real and imaginary parts.
    Complex(f64, f64),
    /// For text.
    Str(String),
}

impl Fill {
    /// The default for a type: false, 0, 0.0 or "".
    #[must_use]
    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool(false),
            DType::F32 | DType::F64 => Self::Float(0.0),
            DType::C64 | DType::C128 => Self::Complex(0.0, 0.0),
            DType::FixedBytes(_) | DType::FixedUnicode(_) | DType::String => {
                Self::Str(String::new())
            }
            _ => Self::Int(0),
        }
    }

    /// Parse a JSON fill value (`null` gives `None`). Floats may be `"NaN"`, `"Infinity"`,
    /// `"-Infinity"` or a hexadecimal bit pattern `"0x7fc00000"`; complex numbers are the
    /// two-element array `[real, imaginary]` of such floats (both Zarr specs).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a value that does not fit the type.
    pub fn parse(dtype: DType, v: &Value) -> Result<Option<Self>, Error> {
        let wrong = || bad(format!("fill value {v} for {dtype:?}"));
        if v.is_null() {
            return Ok(None);
        }
        Ok(Some(match dtype {
            DType::Bool => Self::Bool(match v {
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_i64().ok_or_else(wrong)? != 0,
                _ => return Err(wrong()),
            }),
            DType::F32 | DType::F64 => {
                Self::Float(float_fill(v, dtype == DType::F32).ok_or_else(wrong)?)
            }
            DType::C64 | DType::C128 => match v {
                Value::Array(parts) => match parts.as_slice() {
                    [re, im] => {
                        let narrow = dtype == DType::C64;
                        Self::Complex(
                            float_fill(re, narrow).ok_or_else(wrong)?,
                            float_fill(im, narrow).ok_or_else(wrong)?,
                        )
                    }
                    _ => return Err(wrong()),
                },
                // A real number: zero imaginary part.
                real => Self::Complex(
                    float_fill(real, dtype == DType::C64).ok_or_else(wrong)?,
                    0.0,
                ),
            },
            DType::FixedBytes(_) | DType::FixedUnicode(_) | DType::String => Self::Str(match v {
                Value::String(s) => s.clone(),
                Value::Number(n) if n.as_i64() == Some(0) => String::new(),
                _ => return Err(wrong()),
            }),
            _ => Self::Int(match v {
                Value::Number(n) => n
                    .as_i64()
                    .map(i128::from)
                    .or_else(|| n.as_u64().map(i128::from))
                    .ok_or_else(wrong)?,
                _ => return Err(wrong()),
            }),
        }))
    }

    /// The JSON form (NaN and infinities as strings, as the Zarr specs write them).
    #[must_use]
    pub fn to_json(&self) -> Value {
        match self {
            Self::Bool(b) => Value::Bool(*b),
            Self::Int(v) => i64::try_from(*v)
                .map(Value::from)
                .or_else(|_| u64::try_from(*v).map(Value::from))
                .unwrap_or(Value::Null),
            Self::Float(f) => float_json(*f),
            Self::Complex(re, im) => Value::Array(vec![float_json(*re), float_json(*im)]),
            Self::Str(s) => Value::String(s.clone()),
        }
    }
}

/// A float fill value: a number, `"NaN"`, `"Infinity"`, `"-Infinity"` or a hexadecimal bit
/// pattern (of an `f32` when `narrow`).
fn float_fill(v: &Value, narrow: bool) -> Option<f64> {
    if let Some(f) = crate::json::special_float(v) {
        return Some(f);
    }
    Some(match v {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            hex if hex.starts_with("0x") => {
                let bits = u64::from_str_radix(hex.get(2..).unwrap_or(""), 16).ok()?;
                if narrow {
                    f64::from(f32::from_bits(u32::try_from(bits).ok()?))
                } else {
                    f64::from_bits(bits)
                }
            }
            _ => return None,
        },
        _ => return None,
    })
}

/// A float as the Zarr specs write it: NaN and infinities as strings.
fn float_json(f: f64) -> Value {
    if f.is_nan() {
        Value::String("NaN".into())
    } else if f.is_infinite() {
        Value::String(if f > 0.0 { "Infinity" } else { "-Infinity" }.into())
    } else {
        serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
    }
}
