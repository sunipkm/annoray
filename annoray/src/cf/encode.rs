//! `Dataset` → raw netCDF (the inverse of `decode`), shaped for a target format.

use crate::io::{Format, RawDataset, RawDim, RawVariable, Storage};
use crate::{
    AttrValue, Attrs, DType, DatasetBase, LazySource, NaT, Name, Owned, Repr, TypedArray,
    VariableBase, View,
};
use annoray_units::Duration;
use annoray_units::{Calendar, CfDate, Unit};
use ndarray::{ArrayD, IxDyn};

use crate::cf::num::{attr_f64, attr_of, f64_to_int, flip_sign, is_float, is_int, to_f64};
use crate::cf::{CfError, Result};

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
    enc: &crate::Encoding,
    attrs: &mut Attrs,
    int64_ok: bool,
) -> Result<TypedArray<Owned>> {
    // NaT is skipped when choosing units and written as `i64::MIN` (NaN in floats), as
    // xarray does.
    let real = || dates.iter().filter(|d| !d.is_nat());
    let calendar = real()
        .next()
        .map_or(Calendar::ProlepticGregorian, CfDate::calendar);
    let units = match &enc.time_units {
        Some(u) => u.clone(),
        None => {
            let reference = match real().next() {
                Some(f) => *f,
                None => CfDate::from_ymd(calendar, 1970, 1, 1)?,
            };
            let offsets = real()
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
        .map(|d| {
            if d.is_nat() {
                Ok(i64::MIN)
            } else {
                codec.encode_int(d)
            }
        })
        .collect::<std::result::Result<Vec<i64>, _>>()
    {
        // Without 64-bit integers (netCDF-3), NaT needs floats (NaN), as in xarray.
        Ok(ints)
            if enc.dtype.is_none_or(is_int) && (int64_ok || !dates.iter().any(CfDate::is_nat)) =>
        {
            TypedArray::I64(
                ArrayD::from_shape_vec(shape, ints).map_err(|e| invalid(var, e.to_string()))?,
            )
        }
        // Integers asked for (the file's own type, say) but the times are not whole
        // multiples of the units: an error, as in xarray, not floats.
        Err(e) if enc.dtype.is_some_and(is_int) => {
            return Err(invalid(
                var,
                format!(
                    "the times cannot be stored as {} in {units:?} ({e}); finer units \
                     (or a float dtype) in the encoding would hold them",
                    enc.dtype.map_or_else(String::new, |d| d.to_string())
                ),
            ));
        }
        _ => {
            let floats = dates
                .iter()
                .map(|d| {
                    if d.is_nat() {
                        Ok(f64::NAN)
                    } else {
                        codec.encode_float(d)
                    }
                })
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
    a: &ArrayD<Duration>,
    var: &str,
    attrs: &mut Attrs,
    int64_ok: bool,
) -> Result<TypedArray<Owned>> {
    // NaT is written as `i64::MIN`, as xarray does.
    let ns: Vec<Option<i128>> = a
        .iter()
        .map(|t| {
            if t.is_nat() {
                return Some(None);
            }
            Some(Some(i128::from(t.nanos())))
        })
        .collect::<Option<_>>()
        .ok_or_else(|| invalid(var, "duration out of range"))?;
    let real: Vec<i128> = ns.iter().flatten().copied().collect();
    let (step, step_ns) = coarsest_step(&real);
    attrs.insert("units".into(), AttrValue::from(step));
    // How xarray recognises durations when reading.
    attrs.insert("dtype".into(), AttrValue::from("timedelta64[ns]"));
    if !int64_ok && ns.iter().any(Option::is_none) {
        // Without 64-bit integers (netCDF-3), NaT needs floats (NaN), as in xarray.
        #[allow(clippy::cast_precision_loss)] // whole steps, exact up to 2^53
        let floats: Vec<f64> = ns
            .iter()
            .map(|n| n.map_or(f64::NAN, |n| n.checked_div(step_ns).unwrap_or(0) as f64))
            .collect();
        return Ok(TypedArray::F64(
            ArrayD::from_shape_vec(a.raw_dim(), floats).map_err(|e| invalid(var, e.to_string()))?,
        ));
    }
    let ints = ns
        .iter()
        .map(|n| match n {
            None => Some(i64::MIN),
            Some(n) => n.checked_div(step_ns).and_then(|q| i64::try_from(q).ok()),
        })
        .collect::<Option<Vec<i64>>>()
        .ok_or_else(|| invalid(var, "duration out of range"))?;
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
            return Err(CfError::Io(crate::io::IoError::Format {
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
                    attrs.insert(key.into(), crate::cf::num::flip_sign_attr(v));
                }
            }
            Ok(flip_sign(data))
        }
        DType::I64 | DType::U64 => {
            let f = to_f64(&data).ok_or_else(|| invalid(var, "not numeric"))?;
            f64_to_int(&f, DType::I32, None, var).map_err(|_| {
                CfError::Io(crate::io::IoError::Format {
                    what: format!("variable {var:?}"),
                    format,
                    reason: "64-bit integer values do not fit in 32 bits".into(),
                })
            })
        }
        _ => Ok(data),
    }
}

struct Encoded<'a> {
    raw: RawVariable<'a>,
    char_dim: Option<(Name, usize)>,
}

/// Values on their way to the disk: borrowed from the variable until an encoding step changes
/// them (most variables are written as they are, so most are never copied).
enum Vals<'a> {
    Owned(TypedArray<Owned>),
    Borrowed(TypedArray<View<'a>>),
}

impl<'a> Vals<'a> {
    fn dtype(&self) -> DType {
        match self {
            Self::Owned(a) => a.dtype(),
            Self::Borrowed(a) => a.dtype(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Owned(a) => a.shape(),
            Self::Borrowed(a) => a.shape(),
        }
    }

    /// The values, owned (copied if borrowed), for a step that changes them.
    fn into_owned(self) -> TypedArray<Owned> {
        match self {
            Self::Owned(a) => a,
            Self::Borrowed(a) => a.to_owned(),
        }
    }

    fn into_raw(self) -> crate::io::RawData<'a> {
        match self {
            Self::Owned(a) => a.into(),
            Self::Borrowed(a) => a.into(),
        }
    }
}

/// Where a slab sits in its variable: BitGroom quantization alternates by position in the
/// whole array.
#[derive(Clone, Copy)]
struct At<'b> {
    start: &'b [usize],
    full: &'b [usize],
}

/// `t` quantized as `enc` asks (netCDF-C's algorithms, netCDF4-python's
/// `least_significant_digit`); `fill` values are left as they are.
fn quantized(
    t: TypedArray<Owned>,
    enc: &crate::Encoding,
    fill: Option<f64>,
    at: Option<At<'_>>,
) -> Result<TypedArray<Owned>> {
    let shape = t.shape().to_vec();
    // Odd position of local element `i` in the whole array (C order).
    let odd = move |i: usize| -> bool {
        let Some(At { start, full }) = at else {
            return i % 2 == 1;
        };
        let (mut rest, mut flat, mut stride) = (i, 0usize, 1usize);
        for ((n, s), f) in shape.iter().zip(start).zip(full).rev() {
            let local = rest.checked_rem(*n).unwrap_or(0);
            rest = rest.checked_div(*n).unwrap_or(0);
            flat = flat.wrapping_add(s.wrapping_add(local).wrapping_mul(stride));
            stride = stride.wrapping_mul(*f);
        }
        flat % 2 == 1
    };
    macro_rules! go {
        ($a:expr, $v:ident) => {{
            let mut a = $a.as_standard_layout().into_owned();
            if let Some(values) = a.as_slice_mut() {
                if let Some(q) = enc.quantize {
                    crate::cf::quantize::quantize(values, q, fill, &odd);
                }
                if let Some(d) = enc.least_significant_digit {
                    crate::cf::quantize::least_significant_digit(values, d, fill);
                }
            }
            TypedArray::$v(a)
        }};
    }
    Ok(match t {
        TypedArray::F32(a) => go!(a, F32),
        TypedArray::F64(a) => go!(a, F64),
        other => other,
    })
}

fn encode_var<'a, R: Repr>(
    name: &str,
    v: &'a VariableBase<R>,
    format: Format,
    at: Option<At<'_>>,
) -> Result<Encoded<'a>> {
    let enc = v.encoding();
    let mut attrs = v.attrs().clone();
    let mut dims: Vec<Name> = v.dims().to_vec();
    let view = v.typed()?;
    let mut is_char = false;
    let mut char_dim = None;

    // Units (non-time): the unit's own spelling.
    if let Some(u) = v.unit() {
        attrs.insert("units".into(), AttrValue::Str(u.to_string()));
    }
    // The zone of date-times, as an attribute of annoray's (CF has none).
    if let Some(z) = v.time_zone() {
        match attrs.get("time_zone") {
            Some(AttrValue::Str(a)) if a == z.name() => {}
            Some(_) => {
                return Err(invalid(
                    name,
                    format!("a time_zone attribute differs from the time zone {z}"),
                ));
            }
            None => {
                attrs.insert("time_zone".into(), AttrValue::Str(z.name().to_owned()));
            }
        }
    }
    // CF attributes `decode_coords_all` moved to the encoding.
    for (attr, value) in &enc.cf_related {
        if attrs.contains_key(attr.as_str()) {
            return Err(invalid(
                name,
                format!("{attr:?} is both an attribute and in the encoding"),
            ));
        }
        attrs.insert(attr.as_str().into(), AttrValue::Str(value.clone()));
    }

    let int64_ok = format.supports(DType::I64);
    let was_time = matches!(
        view.dtype(),
        DType::DateTime | DType::CfDate | DType::TimeDelta
    );
    let converted = was_time
        || view.dtype() == DType::Bool
        || (view.dtype() == DType::Str && (format.is_classic_model() || enc.char_dim.is_some()));
    let mut data = if !converted {
        Vals::Borrowed(view)
    } else {
        Vals::Owned(match view.to_owned() {
            TypedArray::DateTime(a) => {
                let cal = enc
                    .calendar
                    .as_deref()
                    .map(str::parse::<Calendar>)
                    .transpose()?
                    .unwrap_or(Calendar::ProlepticGregorian);
                let dates = a
                    .iter()
                    .map(|d| {
                        if d.is_nat() {
                            Ok(CfDate::NAT)
                        } else {
                            d.in_calendar(cal)
                        }
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                encode_times(dates, a.shape(), name, enc, &mut attrs, int64_ok)?
            }
            TypedArray::CfDate(a) => encode_times(
                a.iter().copied().collect(),
                a.shape(),
                name,
                enc,
                &mut attrs,
                int64_ok,
            )?,
            TypedArray::TimeDelta(a) => encode_durations(&a, name, &mut attrs, int64_ok)?,
            TypedArray::Bool(a) => {
                attrs.insert("dtype".into(), AttrValue::from("bool"));
                TypedArray::I8(a.mapv(i8::from))
            }
            // Characters in the classic model, or where the encoding names a character
            // dimension (strings read from a character array go back as one, as in xarray).
            TypedArray::Str(a) if format.is_classic_model() || enc.char_dim.is_some() => {
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
        })
    };

    // Masking and packing back to the on-disk type.
    if is_float(data.dtype()) {
        let fill = enc.fill_value.as_ref().and_then(attr_f64);
        // What NaN becomes: the fill value, else `missing_value` (as xarray masks).
        let nan_as = fill.or_else(|| enc.missing_value.as_ref().and_then(attr_f64));
        // Times already chose their stored type (floats only when NaT needs NaN).
        let target = enc.dtype.filter(|d| is_int(*d) && !was_time);
        let packed = enc.scale_factor.is_some() || enc.add_offset.is_some();
        if packed || target.is_some() {
            let s = enc.scale_factor.as_ref().and_then(attr_f64).unwrap_or(1.0);
            let o = enc.add_offset.as_ref().and_then(attr_f64).unwrap_or(0.0);
            let f = to_f64(&data.into_owned())
                .ok_or_else(|| invalid(name, "not numeric"))?
                .mapv(|x| (x - o) / s);
            let to = target.unwrap_or(DType::I16);
            data = Vals::Owned(f64_to_int(&f, to, nan_as, name)?);
            for (k, a) in [
                ("scale_factor", &enc.scale_factor),
                ("add_offset", &enc.add_offset),
            ] {
                if let Some(a) = a {
                    attrs.insert(k.into(), a.clone());
                }
            }
        } else if let Some(fv) = nan_as.filter(|f| !f.is_nan()) {
            // Replace NaN with a non-NaN fill value.
            data = Vals::Owned(match data.into_owned() {
                TypedArray::F32(a) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let fv32 = fv as f32;
                    TypedArray::F32(a.mapv(|x| if x.is_nan() { fv32 } else { x }))
                }
                TypedArray::F64(a) => TypedArray::F64(a.mapv(|x| if x.is_nan() { fv } else { x })),
                other => other,
            });
        }
        let disk = data.dtype();
        if is_float(disk) && (enc.quantize.is_some() || enc.least_significant_digit.is_some()) {
            data = Vals::Owned(quantized(data.into_owned(), enc, fill, at)?);
            if let Some(q) = enc.quantize {
                attrs.insert(q.attribute().into(), AttrValue::I32(vec![i32::from(q.nsd)]));
            }
            if let Some(d) = enc.least_significant_digit {
                attrs.insert("least_significant_digit".into(), AttrValue::I32(vec![d]));
            }
        }
        match fill {
            Some(f) => {
                if let Some(a) = attr_of(disk, f) {
                    attrs.insert("_FillValue".into(), a);
                }
            }
            // xarray's default for floats (unless `_FillValue: None` asked for none).
            None if is_float(disk) && !enc.no_fill_value => {
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
        let unchanged = format == Format::Netcdf3Data64 || format.supports(data.dtype());
        if !is_char && !unchanged {
            data = Vals::Owned(classic_data(data.into_owned(), name, &mut attrs, format)?);
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
        storage.compression = enc.compression;
        storage.shuffle = enc.shuffle;
        storage.fletcher32 = enc.fletcher32;
        storage.big_endian = enc.big_endian;
        storage.contiguous = enc.contiguous;
        storage.float16 = enc.float16 && is_float(data.dtype());
        storage.ragged.clone_from(&enc.ragged);
        storage.compound.clone_from(&enc.compound);
        storage.enumeration.clone_from(&enc.enumeration);
    }
    Ok(Encoded {
        raw: RawVariable {
            dims,
            data: data.into_raw(),
            is_char,
            attrs,
            storage,
        },
        char_dim,
    })
}

/// Whether `v` can be encoded slice by slice: its encoding must not depend on all of its
/// values (string widths, duration units and unset time units do; and without 64-bit
/// integers the stored type of times depends on whether any is NaT).
fn encodable_lazily<R: Repr>(v: &VariableBase<R>, format: Format) -> bool {
    v.is_lazy()
        && !v.is_empty()
        && match v.dtype() {
            DType::Str | DType::TimeDelta => false,
            DType::DateTime | DType::CfDate => {
                v.encoding().time_units.is_some() && format.supports(DType::I64)
            }
            _ => true,
        }
}

/// Encode `v`, lazily when possible: its attributes and stored type come from encoding its
/// first element, and its values are encoded as they are read.
/// One variable CF-encoded on its own (values, attributes, on-disk dimensions), for writing
/// into an existing file.
pub(crate) fn encode_variable<'a, R: Repr>(
    name: &str,
    v: &'a VariableBase<R>,
    format: Format,
) -> Result<RawVariable<'a>> {
    encode_maybe_lazy(name, v, format).map(|e| e.raw)
}

fn encode_maybe_lazy<'a, R: Repr>(
    name: &str,
    v: &'a VariableBase<R>,
    format: Format,
) -> Result<Encoded<'a>> {
    if !encodable_lazily(v, format) {
        return encode_var(name, v, format, None);
    }
    let first: Vec<(Name, std::ops::Range<usize>)> =
        v.dims().iter().map(|d| (d.clone(), 0..1)).collect();
    let mut probe = v.isel(first)?.to_owned();
    probe.load()?;
    let e = encode_var(name, &probe, format, None)?;
    let RawVariable {
        dims,
        data: probed,
        is_char,
        attrs,
        mut storage,
    } = e.raw;
    // The probe's storage was fitted to one element: fit the chunks to the real shape.
    if storage.chunks.is_some() {
        storage.chunks = v.encoding().chunks.as_ref().map(|c| {
            c.iter()
                .zip(v.shape())
                .map(|(c, n)| (*c).clamp(1, (*n).max(1)))
                .collect()
        });
    }
    let source = EncodeSource {
        inner: v.as_lazy_source()?,
        name: name.to_owned(),
        dims: v.dims().to_vec(),
        unit: v.unit().cloned(),
        attrs: v.attrs().clone(),
        encoding: v.encoding().clone(),
        format,
        dtype: probed.dtype(),
        shape: v.shape().to_vec(),
    };
    Ok(Encoded {
        raw: RawVariable {
            dims,
            data: crate::io::RawData::Lazy(std::sync::Arc::new(source)),
            is_char,
            attrs,
            storage,
        },
        char_dim: e.char_dim,
    })
}

/// A lazy variable encoded slice by slice as it is read (for writing without loading).
#[derive(Debug)]
struct EncodeSource {
    inner: std::sync::Arc<dyn LazySource>,
    name: String,
    dims: Vec<Name>,
    unit: Option<Unit>,
    attrs: Attrs,
    encoding: crate::Encoding,
    format: Format,
    dtype: DType,
    shape: Vec<usize>,
}

impl LazySource for EncodeSource {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> crate::Result<TypedArray<Owned>> {
        let err = |e: CfError| crate::CoreError::Read(e.to_string());
        let mut var = crate::Variable::from_typed(
            self.dims.iter().cloned(),
            self.inner.read_slice(start, count)?,
        )?
        .with_encoding(self.encoding.clone());
        var.set_unit(self.unit.clone());
        var.attrs_mut()
            .extend(self.attrs.iter().map(|(k, a)| (k.clone(), a.clone())));
        let at = At {
            start,
            full: &self.shape,
        };
        let e = encode_var(&self.name, &var, self.format, Some(at)).map_err(err)?;
        let data = e
            .raw
            .data
            .values()
            .map_err(|e| crate::CoreError::Read(e.to_string()))?
            .into_owned();
        if data.dtype() != self.dtype {
            return Err(crate::CoreError::Read(format!(
                "{}: encoded as {} here but {} elsewhere",
                self.name,
                data.dtype(),
                self.dtype
            )));
        }
        Ok(data)
    }
}

/// Encode a dataset for writing in `opts.format`.
///
/// # Errors
/// [`CfError`] if values cannot be represented (e.g. NaN in integer data without a fill value,
/// 64-bit integers too large for a netCDF-3 file).
///
/// Values written unchanged are borrowed from `ds`, not copied.
pub fn encode<'a, S: Repr>(ds: &'a DatasetBase<S>, opts: &EncodeOptions) -> Result<RawDataset<'a>> {
    // Stacked dimensions are written in CF "compression by gathering" form (of a dataset made
    // here, so its values are owned).
    match crate::stack::compress_stacked(ds)? {
        Some(c) => encode_plain(&c, opts).map(RawDataset::into_owned),
        None => encode_plain(ds, opts),
    }
}

fn encode_plain<'a, S: Repr>(
    ds: &'a DatasetBase<S>,
    opts: &EncodeOptions,
) -> Result<RawDataset<'a>> {
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

    // Non-dimension coordinates are listed in each data variable's `coordinates` attribute,
    // except those a CF attribute (cell bounds, grid mapping, …) refers to: they are not
    // coordinates in CF's sense, as xarray has it.
    let referenced: Vec<String> = ds
        .coords()
        .values()
        .map(|c| c.encoding())
        .chain(ds.data_vars().values().map(|v| v.encoding()))
        .flat_map(|e| e.cf_related.iter())
        .flat_map(|(_, value)| {
            value
                .replace(" :", ":")
                .split_whitespace()
                .filter(|w| !w.ends_with(':'))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    let aux: Vec<(&Name, &[Name])> = ds
        .coords()
        .iter()
        .filter(|(k, c)| c.dims() != [k.as_str()] && !referenced.iter().any(|r| r == k.as_str()))
        .map(|(k, c)| (k, c.dims()))
        .collect();
    let mut attached: Vec<&Name> = Vec::new();

    let add = |raw: &mut RawDataset<'a>, name: &Name, e: Encoded<'a>| -> Result<()> {
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
        let e = encode_maybe_lazy(name, c, format)?;
        add(&mut raw, name, e)?;
    }
    for (name, v) in ds.data_vars() {
        let mut e = encode_maybe_lazy(name, v, format)?;
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
