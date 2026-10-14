//! Raw netCDF → `Dataset` (xarray's `decode_cf`).

use std::collections::HashSet;

use chrono::NaiveDateTime;
use ndarray::{ArrayD, Axis, IxDyn};
use xr_core::xr_units::{Calendar, CfDate, Unit, UnitKind};
use xr_core::{AttrValue, DType, Dataset, Encoding, Name, Owned, TypedArray, Variable};
use xr_io::{RawDataset, RawVariable};

use crate::num::{
    attr_dtype, attr_f64, attr_f64s, f64_to_float, flip_sign, flip_sign_attr, is_float, is_int,
    to_f64,
};
use crate::{CfError, Result};

/// What to decode. Everything is on by default, as in xarray.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Turn `_FillValue`/`missing_value` into NaN and apply `scale_factor`/`add_offset`.
    pub mask_and_scale: bool,
    /// Turn `<unit> since <date>` variables into dates.
    pub decode_times: bool,
    /// Parse `units` attributes into [`Unit`]s.
    pub decode_units: bool,
    /// Promote variables named in `coordinates` attributes (and dimension variables) to
    /// coordinates.
    pub decode_coords: bool,
    /// Join `NC_CHAR` arrays into strings along their last dimension.
    pub concat_characters: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            mask_and_scale: true,
            decode_times: true,
            decode_units: true,
            decode_coords: true,
            concat_characters: true,
        }
    }
}

impl DecodeOptions {
    /// Decode nothing: values and attributes exactly as stored.
    #[must_use]
    pub const fn raw() -> Self {
        Self {
            mask_and_scale: false,
            decode_times: false,
            decode_units: false,
            decode_coords: false,
            concat_characters: false,
        }
    }
}

/// A decoded dataset plus anything that could not be decoded.
#[derive(Debug)]
pub struct Decoded {
    /// The dataset.
    pub dataset: Dataset,
    /// Non-fatal problems (unparseable units, undecodable times, skipped variables).
    pub warnings: Vec<String>,
}

fn invalid(var: &str, reason: impl Into<String>) -> CfError {
    CfError::Invalid {
        var: var.into(),
        reason: reason.into(),
    }
}

fn shape_err(var: &str) -> impl Fn(ndarray::ShapeError) -> CfError + '_ {
    move |e| invalid(var, e.to_string())
}

/// Join the last axis of a char array into strings, dropping trailing NULs.
fn chars_to_strings(a: &ArrayD<u8>, var: &str) -> Result<ArrayD<String>> {
    if a.ndim() == 0 {
        let s = a
            .iter()
            .map(|b| char::from(*b))
            .filter(|c| *c != '\0')
            .collect();
        return Ok(ArrayD::from_elem(IxDyn(&[]), s));
    }
    let last = Axis(a.ndim().saturating_sub(1));
    let strings: Vec<String> = a
        .lanes(last)
        .into_iter()
        .map(|lane| {
            let bytes: Vec<u8> = lane.iter().copied().collect();
            let end = bytes
                .iter()
                .rposition(|b| *b != 0)
                .map_or(0, |p| p.saturating_add(1));
            String::from_utf8_lossy(bytes.get(..end).unwrap_or_default()).into_owned()
        })
        .collect();
    let shape: Vec<usize> = a
        .shape()
        .iter()
        .take(a.ndim().saturating_sub(1))
        .copied()
        .collect();
    ArrayD::from_shape_vec(IxDyn(&shape), strings).map_err(shape_err(var))
}

fn str_attr(a: Option<&AttrValue>) -> Option<&str> {
    match a {
        Some(AttrValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Dates from numbers in an epoch unit.
fn decode_times(
    data: &TypedArray<Owned>,
    unit: &Unit,
    calendar: Calendar,
    var: &str,
) -> Result<TypedArray<Owned>> {
    let codec = unit.epoch_codec(calendar)?;
    let dates: Vec<CfDate> = match data {
        TypedArray::F32(a) => a
            .iter()
            .map(|v| codec.decode_float(f64::from(*v)))
            .collect::<std::result::Result<_, _>>()?,
        TypedArray::F64(a) => a
            .iter()
            .map(|v| codec.decode_float(*v))
            .collect::<std::result::Result<_, _>>()?,
        other => {
            let ints = to_f64(other).ok_or_else(|| invalid(var, "time values are not numeric"))?;
            // Integer data: decode exactly.
            match other {
                TypedArray::I64(a) => a
                    .iter()
                    .map(|v| codec.decode_int(*v))
                    .collect::<std::result::Result<_, _>>()?,
                TypedArray::U64(a) => a
                    .iter()
                    .map(|v| {
                        codec.decode_int(
                            i64::try_from(*v)
                                .map_err(|_| xr_core::xr_units::UnitError::Overflow)?,
                        )
                    })
                    .collect::<std::result::Result<_, _>>()?,
                #[allow(clippy::cast_possible_truncation)] // these widen losslessly to i64 via f64
                _ => ints
                    .iter()
                    .map(|v| codec.decode_int(*v as i64))
                    .collect::<std::result::Result<_, _>>()?,
            }
        }
    };
    let shape = data.shape().to_vec();
    if calendar.is_real_world() {
        let chrono: Option<Vec<NaiveDateTime>> = dates
            .iter()
            .map(|d| NaiveDateTime::try_from(*d).ok())
            .collect();
        if let Some(c) = chrono {
            return Ok(TypedArray::DateTime(
                ArrayD::from_shape_vec(IxDyn(&shape), c).map_err(shape_err(var))?,
            ));
        }
    }
    Ok(TypedArray::CfDate(
        ArrayD::from_shape_vec(IxDyn(&shape), dates).map_err(shape_err(var))?,
    ))
}

/// The float type xarray picks for masked or unpacked data.
fn float_dtype(data: DType, scale: Option<&AttrValue>, offset: Option<&AttrValue>) -> DType {
    let small_int = matches!(data, DType::I8 | DType::I16 | DType::U8 | DType::U16);
    if scale.is_some() || offset.is_some() {
        let f32_attrs = [scale, offset]
            .into_iter()
            .flatten()
            .all(|a| attr_dtype(a) == Some(DType::F32));
        if f32_attrs && (small_int || data == DType::F32) {
            DType::F32
        } else {
            DType::F64
        }
    } else if data == DType::F32 || small_int {
        DType::F32
    } else {
        DType::F64
    }
}

fn decode_var(
    name: &str,
    raw: &RawVariable,
    opts: &DecodeOptions,
    warnings: &mut Vec<String>,
) -> Result<Variable> {
    let mut data = raw.data.clone();
    let mut dims = raw.dims.clone();
    let mut attrs = raw.attrs.clone();
    let mut enc = Encoding::default();
    enc.dtype = Some(data.dtype());
    enc.chunks.clone_from(&raw.storage.chunks);
    enc.deflate = raw.storage.deflate;
    enc.shuffle = raw.storage.shuffle;

    // Text stored as characters.
    if raw.is_char && opts.concat_characters {
        if let TypedArray::U8(a) = &data {
            data = TypedArray::Str(chars_to_strings(a, name)?);
            enc.char_dim = dims.pop();
            enc.dtype = Some(DType::Str);
            attrs.shift_remove("_Encoding");
        }
    }

    // `_Unsigned`: signed storage of unsigned values (netCDF-3 has no unsigned types).
    if let Some(flag) = str_attr(attrs.get("_Unsigned")).map(str::to_ascii_lowercase) {
        let signed = matches!(
            data.dtype(),
            DType::I8 | DType::I16 | DType::I32 | DType::I64
        );
        if (flag == "true" && signed) || (flag == "false" && is_int(data.dtype()) && !signed) {
            data = flip_sign(data);
            for key in ["_FillValue", "missing_value", "valid_min", "valid_max"] {
                if let Some(v) = attrs.shift_remove(key) {
                    attrs.insert(key.into(), flip_sign_attr(v));
                }
            }
            enc.unsigned = flag == "true";
        }
        attrs.shift_remove("_Unsigned");
    }

    // Booleans stored as integers (xarray writes `dtype = "bool"`).
    if str_attr(attrs.get("dtype")) == Some("bool") {
        if let Some(f) = to_f64(&data) {
            data = TypedArray::Bool(f.mapv(|v| v != 0.0));
            attrs.shift_remove("dtype");
        }
    }

    // Masking and packing.
    if opts.mask_and_scale && (is_int(data.dtype()) || is_float(data.dtype())) {
        let fill = attrs.shift_remove("_FillValue");
        let missing = attrs.shift_remove("missing_value");
        let scale = attrs.shift_remove("scale_factor");
        let offset = attrs.shift_remove("add_offset");
        let masks: Vec<f64> = fill
            .iter()
            .chain(missing.iter())
            .flat_map(attr_f64s)
            .collect();
        let do_mask = !masks.is_empty();
        let do_scale = scale.is_some() || offset.is_some();
        if do_mask || do_scale {
            let target = float_dtype(data.dtype(), scale.as_ref(), offset.as_ref());
            let s = scale.as_ref().and_then(attr_f64).unwrap_or(1.0);
            let o = offset.as_ref().and_then(attr_f64).unwrap_or(0.0);
            let fill_is_nan = masks.iter().any(|m| m.is_nan());
            let values = to_f64(&data)
                .ok_or_else(|| invalid(name, "not numeric"))?
                .mapv(|v| {
                    if masks.contains(&v) || (fill_is_nan && v.is_nan()) {
                        f64::NAN
                    } else {
                        v.mul_add(s, o)
                    }
                });
            data = f64_to_float(values, target);
        }
        enc.fill_value = fill;
        enc.missing_value = missing;
        enc.scale_factor = scale;
        enc.add_offset = offset;
    }

    let mut unit = None;
    let units_str = str_attr(attrs.get("units")).map(str::to_owned);
    if let Some(units) = units_str {
        match units.parse::<Unit>() {
            Ok(u) if matches!(u.kind(), UnitKind::Epoch(_)) && opts.decode_times => {
                let cal_name = str_attr(attrs.get("calendar")).map(str::to_owned);
                let calendar = match &cal_name {
                    Some(c) => c.parse::<Calendar>(),
                    None => Ok(Calendar::Standard),
                };
                let has_nan = to_f64(&data).is_some_and(|a| a.iter().any(|v| v.is_nan()));
                match (calendar, has_nan) {
                    (Ok(cal), false) => match decode_times(&data, &u, cal, name) {
                        Ok(t) => {
                            data = t;
                            enc.time_units = Some(units.clone());
                            enc.calendar = cal_name;
                            attrs.shift_remove("units");
                            attrs.shift_remove("calendar");
                        }
                        Err(e) => warnings.push(format!("{name}: times left undecoded: {e}")),
                    },
                    (Err(e), _) => warnings.push(format!("{name}: times left undecoded: {e}")),
                    (_, true) => warnings.push(format!(
                        "{name}: times left undecoded: missing values (NaT) are not supported yet"
                    )),
                }
            }
            Ok(u) if opts.decode_units && !matches!(u.kind(), UnitKind::Epoch(_)) => {
                unit = Some(u);
                attrs.shift_remove("units");
            }
            Ok(_) => {}
            Err(e) if opts.decode_units => {
                warnings.push(format!("{name}: units {units:?} kept as an attribute: {e}"))
            }
            Err(_) => {}
        }
    }

    let mut var = Variable::from_typed(dims, data)?.with_encoding(enc);
    if let Some(u) = unit {
        var = var.with_unit(u);
    }
    var.attrs_mut().extend(attrs);
    Ok(var)
}

/// Decode a raw group (use [`RawDataset::group`] to pick a subgroup) into a dataset.
///
/// # Errors
/// [`CfError`] for inconsistent files (dimension lengths, malformed packing). Problems that
/// only affect interpretation (unknown units, undecodable times) become warnings instead.
pub fn decode(raw: &RawDataset, opts: &DecodeOptions) -> Result<Decoded> {
    let mut warnings: Vec<String> = raw
        .skipped
        .iter()
        .map(|(n, why)| format!("{n}: skipped: {why}"))
        .collect();

    // Which variables are coordinates.
    let mut coords: HashSet<String> = HashSet::new();
    if opts.decode_coords {
        for (name, v) in &raw.vars {
            if v.dims.len() == 1 && v.dims.first().is_some_and(|d| d == name) {
                coords.insert(name.to_string());
            }
        }
        let listed = raw
            .vars
            .values()
            .map(|v| &v.attrs)
            .chain(std::iter::once(&raw.attrs));
        for attrs in listed {
            if let Some(AttrValue::Str(list)) = attrs.get("coordinates") {
                coords.extend(
                    list.split_whitespace()
                        .filter(|n| raw.vars.contains_key(*n))
                        .map(str::to_owned),
                );
            }
        }
    }

    let mut ds = Dataset::new();
    let mut data_vars = Vec::new();
    for (name, v) in &raw.vars {
        let mut var = decode_var(name, v, opts, &mut warnings)?;
        if opts.decode_coords {
            var.attrs_mut().shift_remove("coordinates");
        }
        if coords.contains(name.as_str()) {
            ds.add_coord(name.clone(), var)?;
        } else {
            data_vars.push((name.clone(), var));
        }
    }
    for (name, var) in data_vars {
        ds.add_var(name, var)?;
    }
    let mut global = raw.attrs.clone();
    if opts.decode_coords {
        global.shift_remove("coordinates");
    }
    ds.attrs_mut().extend(global);
    ds.encoding_mut().unlimited_dims = raw
        .dims
        .iter()
        .filter(|(_, d)| d.unlimited)
        .map(|(n, _)| Name::clone(n))
        .collect();
    Ok(Decoded {
        dataset: ds,
        warnings,
    })
}
