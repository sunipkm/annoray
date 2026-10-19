//! Numeric conversions used by CF decoding and encoding.

use crate::{AttrValue, DType, Owned, TypedArray};
use ndarray::ArrayD;

use crate::cf::{CfError, Result};

/// The first element of a numeric attribute, as `f64`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn attr_f64(a: &AttrValue) -> Option<f64> {
    match a {
        AttrValue::I8(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::I16(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::I32(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::I64(v) => v.first().map(|x| *x as f64),
        AttrValue::U8(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::U16(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::U32(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::U64(v) => v.first().map(|x| *x as f64),
        AttrValue::F32(v) => v.first().map(|x| f64::from(*x)),
        AttrValue::F64(v) => v.first().copied(),
        _ => None,
    }
}

/// All elements of a numeric attribute, as `f64`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn attr_f64s(a: &AttrValue) -> Vec<f64> {
    match a {
        AttrValue::I8(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::I16(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::I32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::I64(v) => v.iter().map(|x| *x as f64).collect(),
        AttrValue::U8(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::U16(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::U32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::U64(v) => v.iter().map(|x| *x as f64).collect(),
        AttrValue::F32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        AttrValue::F64(v) => v.clone(),
        _ => Vec::new(),
    }
}

pub(crate) fn is_int(d: DType) -> bool {
    matches!(
        d,
        DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
    )
}

pub(crate) fn is_float(d: DType) -> bool {
    matches!(d, DType::F32 | DType::F64)
}

/// Numeric data as `f64` (`None` for non-numeric data).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn to_f64(a: &TypedArray<Owned>) -> Option<ArrayD<f64>> {
    Some(match a {
        TypedArray::F32(x) => x.mapv(f64::from),
        TypedArray::F64(x) => x.clone(),
        TypedArray::I8(x) => x.mapv(f64::from),
        TypedArray::I16(x) => x.mapv(f64::from),
        TypedArray::I32(x) => x.mapv(f64::from),
        TypedArray::I64(x) => x.mapv(|v| v as f64),
        TypedArray::U8(x) => x.mapv(f64::from),
        TypedArray::U16(x) => x.mapv(f64::from),
        TypedArray::U32(x) => x.mapv(f64::from),
        TypedArray::U64(x) => x.mapv(|v| v as f64),
        _ => return None,
    })
}

/// Elements above which [`map_f64`] runs on several threads.
#[cfg(feature = "rayon")]
const PAR_ELEMENTS: usize = 1 << 16;

/// Every element as `f64`, mapped through `f`, in one pass (no `f64` copy first); on several
/// threads for large arrays with the `rayon` feature. `None` for non-numeric data.
#[allow(clippy::cast_precision_loss)] // 64-bit integers are promoted to f64 deliberately
pub(crate) fn map_f64<O: Copy + Send + Sync>(
    a: &TypedArray<Owned>,
    f: impl Fn(f64) -> O + Sync + Send,
) -> Option<ArrayD<O>> {
    macro_rules! go {
        ($x:expr, $conv:expr) => {{
            let zip = ndarray::Zip::from($x);
            let g = |v: &_| f($conv(*v));
            #[cfg(feature = "rayon")]
            if $x.len() > PAR_ELEMENTS {
                return Some(zip.par_map_collect(g));
            }
            Some(zip.map_collect(g))
        }};
    }
    match a {
        TypedArray::F32(x) => go!(x, f64::from),
        TypedArray::F64(x) => go!(x, |v: f64| v),
        TypedArray::I8(x) => go!(x, f64::from),
        TypedArray::I16(x) => go!(x, f64::from),
        TypedArray::I32(x) => go!(x, f64::from),
        TypedArray::I64(x) => go!(x, |v: i64| v as f64),
        TypedArray::U8(x) => go!(x, f64::from),
        TypedArray::U16(x) => go!(x, f64::from),
        TypedArray::U32(x) => go!(x, f64::from),
        TypedArray::U64(x) => go!(x, |v: u64| v as f64),
        _ => None,
    }
}

/// `f64` data rounded into integer type `to`, with NaN replaced by `fill`.
///
/// # Errors
/// [`CfError::Invalid`] if a value (or the fill) is out of range for `to`, or NaN with no fill.
pub(crate) fn f64_to_int(
    a: &ArrayD<f64>,
    to: DType,
    fill: Option<f64>,
    var: &str,
) -> Result<TypedArray<Owned>> {
    let bad = |v: f64| CfError::Invalid {
        var: var.into(),
        reason: format!("value {v} does not fit in {to}"),
    };
    let prep = |v: f64| -> Result<f64> {
        let v = if v.is_nan() {
            fill.ok_or_else(|| CfError::Invalid {
                var: var.into(),
                reason: format!("missing values in {to} data need a _FillValue (or missing_value)"),
            })?
        } else {
            v
        };
        Ok(v.round())
    };
    macro_rules! conv {
        ($var:ident, $t:ty) => {{
            let out = a
                .iter()
                .map(|v| {
                    let r = prep(*v)?;
                    if r >= <$t>::MIN as f64 && r <= <$t>::MAX as f64 {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        Ok(r as $t)
                    } else {
                        Err(bad(r))
                    }
                })
                .collect::<Result<Vec<$t>>>()?;
            TypedArray::$var(ArrayD::from_shape_vec(a.raw_dim(), out).map_err(|e| {
                CfError::Invalid {
                    var: var.into(),
                    reason: e.to_string(),
                }
            })?)
        }};
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(match to {
        DType::I8 => conv!(I8, i8),
        DType::I16 => conv!(I16, i16),
        DType::I32 => conv!(I32, i32),
        DType::I64 => conv!(I64, i64),
        DType::U8 => conv!(U8, u8),
        DType::U16 => conv!(U16, u16),
        DType::U32 => conv!(U32, u32),
        DType::U64 => conv!(U64, u64),
        other => {
            return Err(CfError::Invalid {
                var: var.into(),
                reason: format!("cannot pack into {other}"),
            });
        }
    })
}

/// An attribute holding `v` in element type `d` (for `_FillValue` and friends).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn attr_of(d: DType, v: f64) -> Option<AttrValue> {
    let fits = |lo: f64, hi: f64| v.is_finite() && v >= lo && v <= hi && v.fract() == 0.0;
    Some(match d {
        DType::F32 => AttrValue::F32(vec![v as f32]),
        DType::F64 => AttrValue::F64(vec![v]),
        DType::I8 if fits(-128.0, 127.0) => AttrValue::I8(vec![v as i8]),
        DType::I16 if fits(-32768.0, 32767.0) => AttrValue::I16(vec![v as i16]),
        DType::I32 if fits(-2_147_483_648.0, 2_147_483_647.0) => AttrValue::I32(vec![v as i32]),
        DType::I64 if fits(-9.223_372_036_854_776e18, 9.223_372_036_854_775e18) => {
            AttrValue::I64(vec![v as i64])
        }
        DType::U8 if fits(0.0, 255.0) => AttrValue::U8(vec![v as u8]),
        DType::U16 if fits(0.0, 65535.0) => AttrValue::U16(vec![v as u16]),
        DType::U32 if fits(0.0, 4_294_967_295.0) => AttrValue::U32(vec![v as u32]),
        DType::U64 if fits(0.0, 1.844_674_407_370_955e19) => AttrValue::U64(vec![v as u64]),
        _ => return None,
    })
}

/// Reinterpret signed integers as unsigned of the same width, or the reverse (`_Unsigned`).
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
pub(crate) fn flip_sign(a: TypedArray<Owned>) -> TypedArray<Owned> {
    match a {
        TypedArray::I8(x) => TypedArray::U8(x.mapv(|v| v as u8)),
        TypedArray::I16(x) => TypedArray::U16(x.mapv(|v| v as u16)),
        TypedArray::I32(x) => TypedArray::U32(x.mapv(|v| v as u32)),
        TypedArray::I64(x) => TypedArray::U64(x.mapv(|v| v as u64)),
        TypedArray::U8(x) => TypedArray::I8(x.mapv(|v| v as i8)),
        TypedArray::U16(x) => TypedArray::I16(x.mapv(|v| v as i16)),
        TypedArray::U32(x) => TypedArray::I32(x.mapv(|v| v as i32)),
        TypedArray::U64(x) => TypedArray::I64(x.mapv(|v| v as i64)),
        other => other,
    }
}

/// The same reinterpretation for an attribute value.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
pub(crate) fn flip_sign_attr(a: AttrValue) -> AttrValue {
    match a {
        AttrValue::I8(x) => AttrValue::U8(x.into_iter().map(|v| v as u8).collect()),
        AttrValue::I16(x) => AttrValue::U16(x.into_iter().map(|v| v as u16).collect()),
        AttrValue::I32(x) => AttrValue::U32(x.into_iter().map(|v| v as u32).collect()),
        AttrValue::I64(x) => AttrValue::U64(x.into_iter().map(|v| v as u64).collect()),
        AttrValue::U8(x) => AttrValue::I8(x.into_iter().map(|v| v as i8).collect()),
        AttrValue::U16(x) => AttrValue::I16(x.into_iter().map(|v| v as i16).collect()),
        AttrValue::U32(x) => AttrValue::I32(x.into_iter().map(|v| v as i32).collect()),
        AttrValue::U64(x) => AttrValue::I64(x.into_iter().map(|v| v as i64).collect()),
        other => other,
    }
}

/// The element type of a numeric attribute.
pub(crate) fn attr_dtype(a: &AttrValue) -> Option<DType> {
    Some(match a {
        AttrValue::I8(_) => DType::I8,
        AttrValue::I16(_) => DType::I16,
        AttrValue::I32(_) => DType::I32,
        AttrValue::I64(_) => DType::I64,
        AttrValue::U8(_) => DType::U8,
        AttrValue::U16(_) => DType::U16,
        AttrValue::U32(_) => DType::U32,
        AttrValue::U64(_) => DType::U64,
        AttrValue::F32(_) => DType::F32,
        AttrValue::F64(_) => DType::F64,
        _ => return None,
    })
}
