//! `Dataset` → raw netCDF (the inverse of `decode`), shaped for a target format.

use chrono::{NaiveDateTime, TimeDelta};
use ndarray::{ArrayD, IxDyn};
use xr_core::xr_units::{Calendar, CfDate, Unit};
use xr_core::{AttrValue, Attrs, DType, DatasetBase, Name, Owned, Repr, TypedArray, VariableBase};
use xr_io::{Format, RawDataset, RawDim, RawVariable, Storage};

use crate::num::{attr_f64, attr_of, f64_to_int, flip_sign, is_float, is_int, to_f64};
use crate::{CfError, Result};

/// Options for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// The target format; decides which types and features are available.
    pub format: Format,
}

impl EncodeOptions {
    /// Options for a format.
    #[must_use]
    pub const fn format(format: Format) -> Self {
        Self { format }
    }
}

fn invalid(var: &str, reason: impl Into<String>) -> CfError {
    CfError::Invalid {
        var: var.into(),
        reason: reason.into(),
    }
}

const STEPS: [(&str, i128); 7] = [
    ("days", 86_400_000_000_000),
    ("hours", 3_600_000_000_000),
    ("minutes", 60_000_000_000),
    ("seconds", 1_000_000_000),
    ("milliseconds", 1_000_000),
    ("microseconds", 1_000),
    ("nanoseconds", 1),
];

/// The coarsest step that divides every offset exactly (xarray's `infer_datetime_units`).
fn coarsest_step(offsets: &[i128]) -> (&'static str, i128) {
    STEPS
        .into_iter()
        .find(|(_, ns)| offsets.iter().all(|o| o.checked_rem(*ns) == Some(0)))
        .unwrap_or(("nanoseconds", 1))
}

fn format_ref(d: &CfDate) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        d.year(),
        d.month(),
        d.day(),
        d.hour(),
        d.minute(),
        d.second()
    )
}

/// Encode dates as numbers, choosing units and calendar like xarray when the encoding has none.
fn encode_times(
    dates: Vec<CfDate>,
    shape: &[usize],
    var: &str,
    enc: &xr_core::Encoding,
    attrs: &mut Attrs,
) -> Result<TypedArray<Owned>> {
    let calendar = dates
        .first()
        .map_or(Calendar::ProlepticGregorian, CfDate::calendar);
    let units = match &enc.time_units {
        Some(u) => u.clone(),
        None => {
            let first = dates.first().copied();
            let reference = match first {
                Some(f) => f,
                None => CfDate::from_ymd(calendar, 1970, 1, 1)?,
            };
            let offsets = dates
                .iter()
                .map(|d| d.nanos_since(&reference))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let (step, _) = coarsest_step(&offsets);
            format!("{step} since {}", format_ref(&reference))
        }
    };
    let codec = units.parse::<Unit>()?.epoch_codec(calendar)?;
    let shape = IxDyn(shape);
    let out = match dates
        .iter()
        .map(|d| codec.encode_int(d))
        .collect::<std::result::Result<Vec<i64>, _>>()
    {
        Ok(ints) if enc.dtype.is_none_or(is_int) => TypedArray::I64(
            ArrayD::from_shape_vec(shape, ints).map_err(|e| invalid(var, e.to_string()))?,
        ),
        _ => {
            let floats = dates
                .iter()
                .map(|d| codec.encode_float(d))
                .collect::<std::result::Result<Vec<f64>, _>>()?;
            TypedArray::F64(
                ArrayD::from_shape_vec(shape, floats).map_err(|e| invalid(var, e.to_string()))?,
            )
        }
    };
    attrs.insert("units".into(), AttrValue::Str(units));
    attrs.insert(
        "calendar".into(),
        AttrValue::Str(
            enc.calendar
                .clone()
                .unwrap_or_else(|| calendar.name().to_owned()),
        ),
    );
    Ok(out)
}

/// Durations as integers in the coarsest exact unit.
fn encode_durations(
    a: &ArrayD<TimeDelta>,
    var: &str,
    attrs: &mut Attrs,
) -> Result<TypedArray<Owned>> {
    let ns: Vec<i128> = a
        .iter()
        .map(|t| {
            i128::from(t.num_seconds())
                .checked_mul(1_000_000_000)
                .and_then(|s| s.checked_add(i128::from(t.subsec_nanos())))
        })
        .collect::<Option<_>>()
        .ok_or_else(|| invalid(var, "duration out of range"))?;
    let (step, step_ns) = coarsest_step(&ns);
    let ints = ns
        .iter()
        .map(|n| n.checked_div(step_ns).and_then(|q| i64::try_from(q).ok()))
        .collect::<Option<Vec<i64>>>()
        .ok_or_else(|| invalid(var, "duration out of range"))?;
    attrs.insert("units".into(), AttrValue::from(step));
    Ok(TypedArray::I64(
        ArrayD::from_shape_vec(a.raw_dim(), ints).map_err(|e| invalid(var, e.to_string()))?,
    ))
}

/// Strings as an `NC_CHAR` array with a trailing string-length dimension.
fn strings_to_chars(a: &ArrayD<String>, var: &str) -> Result<(ArrayD<u8>, usize)> {
    let width = a.iter().map(String::len).max().unwrap_or(0).max(1);
    let mut bytes = Vec::with_capacity(a.len().saturating_mul(width));
    for s in a {
        bytes.extend(s.bytes());
        bytes.extend(std::iter::repeat_n(0u8, width.saturating_sub(s.len())));
    }
    let mut shape = a.shape().to_vec();
    shape.push(width);
    Ok((
        ArrayD::from_shape_vec(IxDyn(&shape), bytes).map_err(|e| invalid(var, e.to_string()))?,
        width,
    ))
}

/// Attribute values the classic formats cannot hold, converted losslessly where possible.
#[allow(clippy::cast_precision_loss)]
fn classic_attr(v: &AttrValue, name: &str, format: Format) -> Result<AttrValue> {
    let fits_i32 = |x: i64| i32::try_from(x).is_ok();
    Ok(match v {
        AttrValue::U8(x) => AttrValue::I16(x.iter().map(|v| i16::from(*v)).collect()),
        AttrValue::U16(x) => AttrValue::I32(x.iter().map(|v| i32::from(*v)).collect()),
        AttrValue::U32(x) => AttrValue::F64(x.iter().map(|v| f64::from(*v)).collect()),
        AttrValue::I64(x) if x.iter().all(|v| fits_i32(*v)) => {
            AttrValue::I32(x.iter().filter_map(|v| i32::try_from(*v).ok()).collect())
        }
        AttrValue::I64(x) => AttrValue::F64(x.iter().map(|v| *v as f64).collect()),
        AttrValue::U64(x) => AttrValue::F64(x.iter().map(|v| *v as f64).collect()),
        AttrValue::Strs(x) if x.len() == 1 => {
            AttrValue::Str(x.first().cloned().unwrap_or_default())
        }
        AttrValue::Strs(_) => {
            return Err(CfError::Io(xr_io::IoError::Format {
                what: format!("attribute {name:?}"),
                format,
                reason: "string lists need netCDF-4".into(),
            }));
        }
        other => other.clone(),
    })
}

/// Integers the classic formats lack: unsigned → signed with `_Unsigned`, 64-bit → 32-bit if
/// every value fits.
fn classic_data(
    data: TypedArray<Owned>,
    var: &str,
    attrs: &mut Attrs,
    format: Format,
) -> Result<TypedArray<Owned>> {
    if format == Format::Netcdf3Data64 || format.supports(data.dtype()) {
        return Ok(data);
    }
    match data.dtype() {
        DType::U8 | DType::U16 | DType::U32 => {
            attrs.insert("_Unsigned".into(), AttrValue::from("true"));
            for key in ["_FillValue", "missing_value"] {
                if let Some(v) = attrs.shift_remove(key) {
                    attrs.insert(key.into(), crate::num::flip_sign_attr(v));
                }
            }
            Ok(flip_sign(data))
        }
        DType::I64 | DType::U64 => {
            let f = to_f64(&data).ok_or_else(|| invalid(var, "not numeric"))?;
            f64_to_int(&f, DType::I32, None, var).map_err(|_| {
                CfError::Io(xr_io::IoError::Format {
                    what: format!("variable {var:?}"),
                    format,
                    reason: "64-bit integer values do not fit in 32 bits".into(),
                })
            })
        }
        _ => Ok(data),
    }
}

struct Encoded {
    raw: RawVariable,
    char_dim: Option<(Name, usize)>,
}

fn encode_var<R: Repr>(name: &str, v: &VariableBase<R>, format: Format) -> Result<Encoded> {
    let enc = v.encoding();
    let mut attrs = v.attrs().clone();
    let mut dims: Vec<Name> = v.dims().to_vec();
    let mut data = v.typed().to_owned();
    let mut is_char = false;
    let mut char_dim = None;

    // Units (non-time): the unit's own spelling.
    if let Some(u) = v.unit() {
        attrs.insert("units".into(), AttrValue::Str(u.to_string()));
    }

    data = match data {
        TypedArray::DateTime(a) => {
            let cal = enc
                .calendar
                .as_deref()
                .map(str::parse::<Calendar>)
                .transpose()?
                .unwrap_or(Calendar::ProlepticGregorian);
            let dates = a
                .iter()
                .map(|d: &NaiveDateTime| CfDate::from_chrono(*d, cal))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            encode_times(dates, a.shape(), name, enc, &mut attrs)?
        }
        TypedArray::CfDate(a) => encode_times(
            a.iter().copied().collect(),
            a.shape(),
            name,
            enc,
            &mut attrs,
        )?,
        TypedArray::TimeDelta(a) => encode_durations(&a, name, &mut attrs)?,
        TypedArray::Bool(a) => {
            attrs.insert("dtype".into(), AttrValue::from("bool"));
            TypedArray::I8(a.mapv(i8::from))
        }
        TypedArray::Str(a) if format.is_classic_model() => {
            let (chars, width) = strings_to_chars(&a, name)?;
            let dim = enc
                .char_dim
                .clone()
                .unwrap_or_else(|| Name::from(format!("string{width}")));
            dims.push(dim.clone());
            char_dim = Some((dim, width));
            is_char = true;
            // Tells xarray (and us) the bytes are UTF-8 text, not raw bytes.
            attrs.insert("_Encoding".into(), AttrValue::from("utf-8"));
            TypedArray::U8(chars)
        }
        other => other,
    };

    // Masking and packing back to the on-disk type.
    if is_float(data.dtype()) {
        let fill = enc.fill_value.as_ref().and_then(attr_f64);
        let target = enc.dtype.filter(|d| is_int(*d));
        let packed = enc.scale_factor.is_some() || enc.add_offset.is_some();
        if packed || target.is_some() {
            let s = enc.scale_factor.as_ref().and_then(attr_f64).unwrap_or(1.0);
            let o = enc.add_offset.as_ref().and_then(attr_f64).unwrap_or(0.0);
            let f = to_f64(&data)
                .ok_or_else(|| invalid(name, "not numeric"))?
                .mapv(|x| (x - o) / s);
            let to = target.unwrap_or(DType::I16);
            data = f64_to_int(&f, to, fill, name)?;
            for (k, a) in [
                ("scale_factor", &enc.scale_factor),
                ("add_offset", &enc.add_offset),
            ] {
                if let Some(a) = a {
                    attrs.insert(k.into(), a.clone());
                }
            }
        } else if let Some(fv) = fill.filter(|f| !f.is_nan()) {
            // Replace NaN with a non-NaN fill value.
            data = match data {
                TypedArray::F32(a) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let fv32 = fv as f32;
                    TypedArray::F32(a.mapv(|x| if x.is_nan() { fv32 } else { x }))
                }
                TypedArray::F64(a) => TypedArray::F64(a.mapv(|x| if x.is_nan() { fv } else { x })),
                other => other,
            };
        }
        let disk = data.dtype();
        match fill {
            Some(f) => {
                if let Some(a) = attr_of(disk, f) {
                    attrs.insert("_FillValue".into(), a);
                }
            }
            // xarray's default for floats.
            None if is_float(disk) => {
                if let Some(a) = attr_of(disk, f64::NAN) {
                    attrs.insert("_FillValue".into(), a);
                }
            }
            None => {}
        }
    } else if let Some(f) = &enc.fill_value {
        attrs.insert("_FillValue".into(), f.clone());
    }
    if let Some(m) = &enc.missing_value {
        attrs.insert("missing_value".into(), m.clone());
    }

    if format.is_classic_model() {
        if !is_char {
            data = classic_data(data, name, &mut attrs, format)?;
        }
        attrs = attrs
            .iter()
            .map(|(k, a)| Ok((k.clone(), classic_attr(a, k, format)?)))
            .collect::<Result<_>>()?;
    }

    let mut storage = Storage::default();
    if !format.is_netcdf3() && !dims.is_empty() {
        storage.chunks = enc
            .chunks
            .as_ref()
            .filter(|c| c.len() == dims.len())
            .map(|c| {
                c.iter()
                    .zip(data.shape())
                    .map(|(c, n)| (*c).clamp(1, (*n).max(1)))
                    .collect()
            });
        storage.deflate = enc.deflate;
        storage.shuffle = enc.shuffle;
    }
    Ok(Encoded {
        raw: RawVariable {
            dims,
            data,
            is_char,
            attrs,
            storage,
        },
        char_dim,
    })
}

/// Encode a dataset for writing in `opts.format`.
///
/// # Errors
/// [`CfError`] if values cannot be represented (e.g. NaN in integer data without a fill value,
/// 64-bit integers too large for a netCDF-3 file).
pub fn encode<S: Repr>(ds: &DatasetBase<S>, opts: &EncodeOptions) -> Result<RawDataset> {
    let format = opts.format;
    let mut raw = RawDataset::default();
    let unlimited = &ds.encoding().unlimited_dims;
    for (d, n) in ds.sizes() {
        raw.dims.insert(
            d.clone(),
            RawDim {
                len: n,
                unlimited: unlimited.contains(&d),
            },
        );
    }

    // Non-dimension coordinates are listed in each data variable's `coordinates` attribute.
    let aux: Vec<(&Name, &[Name])> = ds
        .coords()
        .iter()
        .filter(|(k, c)| c.dims() != [k.as_str()])
        .map(|(k, c)| (k, c.dims()))
        .collect();
    let mut attached: Vec<&Name> = Vec::new();

    let add = |raw: &mut RawDataset, name: &Name, e: Encoded| -> Result<()> {
        if let Some((dim, width)) = e.char_dim {
            match raw.dims.get(&dim) {
                Some(existing) if existing.len != width => {
                    return Err(invalid(
                        name,
                        format!("char dimension {dim:?} already has length {}", existing.len),
                    ));
                }
                Some(_) => {}
                None => {
                    raw.dims.insert(
                        dim,
                        RawDim {
                            len: width,
                            unlimited: false,
                        },
                    );
                }
            }
        }
        raw.vars.insert(name.clone(), e.raw);
        Ok(())
    };

    for (name, c) in ds.coords() {
        let e = encode_var(name, c, format)?;
        add(&mut raw, name, e)?;
    }
    for (name, v) in ds.data_vars() {
        let mut e = encode_var(name, v, format)?;
        let mine: Vec<&str> = aux
            .iter()
            .filter(|(_, cd)| cd.iter().all(|d| v.dims().contains(d)))
            .map(|(k, _)| k.as_str())
            .collect();
        if !mine.is_empty() {
            e.raw
                .attrs
                .insert("coordinates".into(), AttrValue::Str(mine.join(" ")));
            attached.extend(
                aux.iter()
                    .filter(|(k, _)| mine.contains(&k.as_str()))
                    .map(|(k, _)| *k),
            );
        }
        add(&mut raw, name, e)?;
    }
    let mut global: Attrs = ds.attrs().clone();
    let orphans: Vec<&str> = aux
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| !attached.contains(k))
        .map(Name::as_str)
        .collect();
    if !orphans.is_empty() {
        global.insert("coordinates".into(), AttrValue::Str(orphans.join(" ")));
    }
    if format.is_classic_model() {
        global = global
            .iter()
            .map(|(k, a)| Ok((k.clone(), classic_attr(a, k, format)?)))
            .collect::<Result<_>>()?;
    }
    raw.attrs = global;
    Ok(raw)
}
