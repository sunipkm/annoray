//! Raw netCDF → `Dataset` (xarray's `decode_cf`).

use std::collections::HashSet;
use std::sync::Arc;

use chrono::NaiveDateTime;
use ndarray::{ArrayD, Axis, IxDyn};
use xr_core::xr_units::{Calendar, CfDate, Unit, UnitKind};
use xr_core::{
    AttrValue, DType, Dataset, Encoding, LazyArray, LazySource, Name, Owned, TypedArray, Variable,
};
use xr_io::{RawData, RawDataset, RawVariable};

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
    /// Keep data variables lazy when the backend provides lazy data (read on first access,
    /// only the selected part).
    pub lazy: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            mask_and_scale: true,
            decode_times: true,
            decode_units: true,
            decode_coords: true,
            concat_characters: true,
            lazy: true,
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
            lazy: true,
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

/// One decoding step, applied to raw values in order. Every step works on any slice of the
/// variable (character joining needs the whole string-length axis, which slices always span).
#[derive(Debug, Clone)]
enum Step {
    Chars,
    FlipSign,
    Bool,
    MaskScale {
        masks: Vec<f64>,
        scale: f64,
        offset: f64,
        target: DType,
    },
}

/// How to decode a variable, worked out from its attributes and on-disk dtype alone.
struct Plan {
    steps: Vec<Step>,
    times: Option<(Unit, Calendar, String, Option<String>)>,
    dims: Vec<Name>,
    attrs: xr_core::Attrs,
    enc: Encoding,
    unit: Option<Unit>,
    dtype: DType,
}

fn flipped(d: DType) -> DType {
    match d {
        DType::I8 => DType::U8,
        DType::I16 => DType::U16,
        DType::I32 => DType::U32,
        DType::I64 => DType::U64,
        DType::U8 => DType::I8,
        DType::U16 => DType::I16,
        DType::U32 => DType::I32,
        DType::U64 => DType::I64,
        other => other,
    }
}

fn plan(name: &str, raw: &RawVariable, opts: &DecodeOptions, warnings: &mut Vec<String>) -> Plan {
    let mut dtype = raw.data.dtype();
    let mut dims = raw.dims.clone();
    let mut attrs = raw.attrs.clone();
    let mut enc = Encoding::default();
    enc.dtype = Some(dtype);
    enc.chunks.clone_from(&raw.storage.chunks);
    enc.deflate = raw.storage.deflate;
    enc.shuffle = raw.storage.shuffle;
    let mut steps = Vec::new();

    // Text stored as characters.
    if raw.is_char && opts.concat_characters && dtype == DType::U8 {
        steps.push(Step::Chars);
        enc.char_dim = dims.pop();
        enc.dtype = Some(DType::Str);
        dtype = DType::Str;
        attrs.shift_remove("_Encoding");
    }

    // `_Unsigned`: signed storage of unsigned values (netCDF-3 has no unsigned types).
    if let Some(flag) = str_attr(attrs.get("_Unsigned")).map(str::to_ascii_lowercase) {
        let signed = matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64);
        if (flag == "true" && signed) || (flag == "false" && is_int(dtype) && !signed) {
            steps.push(Step::FlipSign);
            dtype = flipped(dtype);
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
    if str_attr(attrs.get("dtype")) == Some("bool") && (is_int(dtype) || is_float(dtype)) {
        steps.push(Step::Bool);
        dtype = DType::Bool;
        attrs.shift_remove("dtype");
    }

    // Masking and packing.
    if opts.mask_and_scale && (is_int(dtype) || is_float(dtype)) {
        let fill = attrs.shift_remove("_FillValue");
        let missing = attrs.shift_remove("missing_value");
        let scale = attrs.shift_remove("scale_factor");
        let offset = attrs.shift_remove("add_offset");
        let masks: Vec<f64> = fill
            .iter()
            .chain(missing.iter())
            .flat_map(attr_f64s)
            .collect();
        if !masks.is_empty() || scale.is_some() || offset.is_some() {
            let target = float_dtype(dtype, scale.as_ref(), offset.as_ref());
            steps.push(Step::MaskScale {
                masks,
                scale: scale.as_ref().and_then(attr_f64).unwrap_or(1.0),
                offset: offset.as_ref().and_then(attr_f64).unwrap_or(0.0),
                target,
            });
            dtype = target;
        }
        enc.fill_value = fill;
        enc.missing_value = missing;
        enc.scale_factor = scale;
        enc.add_offset = offset;
    }

    let mut unit = None;
    let mut times = None;
    if let Some(units) = str_attr(attrs.get("units")).map(str::to_owned) {
        match units.parse::<Unit>() {
            Ok(u) if matches!(u.kind(), UnitKind::Epoch(_)) && opts.decode_times => {
                let cal_name = str_attr(attrs.get("calendar")).map(str::to_owned);
                match cal_name
                    .as_deref()
                    .map_or(Ok(Calendar::Standard), str::parse::<Calendar>)
                {
                    Ok(cal) => times = Some((u, cal, units, cal_name)),
                    Err(e) => warnings.push(format!("{name}: times left undecoded: {e}")),
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
    Plan {
        steps,
        times,
        dims,
        attrs,
        enc,
        unit,
        dtype,
    }
}

fn apply(steps: &[Step], mut data: TypedArray<Owned>, name: &str) -> Result<TypedArray<Owned>> {
    for step in steps {
        data = match step {
            Step::Chars => match &data {
                TypedArray::U8(a) => TypedArray::Str(chars_to_strings(a, name)?),
                _ => data,
            },
            Step::FlipSign => flip_sign(data),
            Step::Bool => TypedArray::Bool(
                to_f64(&data)
                    .ok_or_else(|| invalid(name, "not numeric"))?
                    .mapv(|v| v != 0.0),
            ),
            Step::MaskScale {
                masks,
                scale,
                offset,
                target,
            } => {
                let fill_is_nan = masks.iter().any(|m| m.is_nan());
                let values = to_f64(&data)
                    .ok_or_else(|| invalid(name, "not numeric"))?
                    .mapv(|v| {
                        if masks.contains(&v) || (fill_is_nan && v.is_nan()) {
                            f64::NAN
                        } else {
                            v.mul_add(*scale, *offset)
                        }
                    });
                f64_to_float(values, *target)
            }
        };
    }
    Ok(data)
}

/// Time decoding with a fixed output type (needed for lazy data): chrono datetimes for
/// real-world calendars, `CfDate` otherwise. Missing values (NaN) are an error.
fn decode_times_strict(
    data: &TypedArray<Owned>,
    unit: &Unit,
    cal: Calendar,
    name: &str,
) -> Result<TypedArray<Owned>> {
    if to_f64(data).is_some_and(|a| a.iter().any(|v| v.is_nan())) {
        return Err(invalid(name, "missing times (NaT) are not supported yet"));
    }
    let decoded = decode_times(data, unit, cal, name)?;
    match (cal.is_real_world(), decoded) {
        (true, TypedArray::CfDate(_)) => Err(invalid(name, "dates outside chrono's range")),
        (_, d) => Ok(d),
    }
}

/// A lazily decoded variable: reads raw slices and decodes them.
#[derive(Debug)]
struct CfSource {
    inner: Arc<dyn LazySource>,
    steps: Vec<Step>,
    times: Option<(Unit, Calendar)>,
    chars: bool,
    dtype: DType,
    shape: Vec<usize>,
    name: String,
}

impl LazySource for CfSource {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> xr_core::Result<TypedArray<Owned>> {
        let err = |e: CfError| xr_core::CoreError::Read(e.to_string());
        let (mut start, mut count) = (start.to_vec(), count.to_vec());
        if self.chars {
            // Strings need their whole string-length axis.
            start.push(0);
            count.push(self.inner.shape().last().copied().unwrap_or(0));
        }
        let raw = self.inner.read_slice(&start, &count)?;
        let data = apply(&self.steps, raw, &self.name).map_err(err)?;
        match &self.times {
            Some((u, cal)) => decode_times_strict(&data, u, *cal, &self.name).map_err(err),
            None => Ok(data),
        }
    }
}

fn decode_var(
    name: &str,
    raw: &RawVariable,
    opts: &DecodeOptions,
    warnings: &mut Vec<String>,
) -> Result<Variable> {
    let mut p = plan(name, raw, opts, warnings);
    // Dimension coordinates are read eagerly (selection needs them); the rest may stay lazy.
    let index = p.dims.len() == 1 && p.dims.first().is_some_and(|d| d.as_str() == name);
    let var = match (&raw.data, opts.lazy && !index) {
        (RawData::Lazy(src), true) => {
            let mut times = None;
            let mut dtype = p.dtype;
            if let Some((u, cal, units, cal_name)) = p.times.take() {
                dtype = if cal.is_real_world() {
                    DType::DateTime
                } else {
                    DType::CfDate
                };
                times = Some((u, cal));
                p.enc.time_units = Some(units);
                p.enc.calendar = cal_name;
                p.attrs.shift_remove("units");
                p.attrs.shift_remove("calendar");
            }
            let lazy = if p.steps.is_empty() && times.is_none() {
                LazyArray::new(src.clone())
            } else {
                LazyArray::new(Arc::new(CfSource {
                    inner: src.clone(),
                    chars: p.steps.iter().any(|s| matches!(s, Step::Chars)),
                    steps: p.steps.clone(),
                    times,
                    dtype,
                    shape: src.shape().iter().take(p.dims.len()).copied().collect(),
                    name: name.to_owned(),
                }))
            };
            Variable::from_lazy(p.dims.clone(), lazy)?
        }
        _ => {
            let mut data = apply(&p.steps, raw.data.values()?.into_owned(), name)?;
            if let Some((u, cal, units, cal_name)) = p.times.take() {
                let has_nan = to_f64(&data).is_some_and(|a| a.iter().any(|v| v.is_nan()));
                if has_nan {
                    warnings.push(format!(
                        "{name}: times left undecoded: missing values (NaT) are not supported yet"
                    ));
                } else {
                    match decode_times(&data, &u, cal, name) {
                        Ok(t) => {
                            data = t;
                            p.enc.time_units = Some(units);
                            p.enc.calendar = cal_name;
                            p.attrs.shift_remove("units");
                            p.attrs.shift_remove("calendar");
                        }
                        Err(e) => warnings.push(format!("{name}: times left undecoded: {e}")),
                    }
                }
            }
            Variable::from_typed(p.dims.clone(), data)?
        }
    };
    let mut var = var.with_encoding(p.enc);
    if let Some(u) = p.unit {
        var = var.with_unit(u);
    }
    var.attrs_mut().extend(p.attrs);
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
