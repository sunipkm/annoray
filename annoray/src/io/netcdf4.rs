//! Pure-Rust netCDF-4: the netCDF data model on top of HDF5 ([`annoray_hdf5`]), read and
//! written the way netCDF-C does.
//!
//! - **Dimensions** are HDF5 dimension scales (`CLASS = "DIMENSION_SCALE"`). A dimension with no
//!   coordinate variable is an empty dataset whose `NAME` is
//!   `"This is a netCDF dimension but not a netCDF variable.%10d"`. Dimension ids live in
//!   `_Netcdf4Dimid`; unlimited dimensions are scales with an unlimited maximum size.
//! - **Variables** point at their dimensions through `DIMENSION_LIST` (object references) and
//!   carry `_Netcdf4Coordinates` (their dimension ids); each scale lists its users in
//!   `REFERENCE_LIST`. A variable that shares a dimension's name without being its coordinate
//!   variable is stored as `_nc4_non_coord_<name>`.
//! - **Types**: `NC_CHAR` is a 1-byte fixed string; `NC_STRING` is a variable-length string;
//!   text attributes are exact-length fixed strings.
//! - **`_FillValue`** is both the dataset fill property and an ordinary attribute.
//! - The root group carries `_NCProperties`; the classic model adds `_nc3_strict`.
//!
//! These bookkeeping attributes are hidden when reading. HDF5 files without them (plain HDF5)
//! get netCDF-C's `phony_dim_N` dimensions.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use annoray_hdf5::{
    Array, AttrData, Dataset, Datatype, File, Group, Node, ValueType, WDataset, WGroup, WNode,
};
use ndarray::{ArrayD, IxDyn};

use crate::io::{
    Backend, Format, IoError, RawData, RawDataset, RawDim, RawVariable, Result, Storage,
    WriteOptions,
};
use crate::{AttrValue, Attrs, CoreError, DType, LazySource, Name, Owned, TypedArray};

/// The pure-Rust netCDF-4 backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Netcdf4;

const NAME: &str = "netcdf4";
pub(crate) const NON_COORD: &str = "_nc4_non_coord_";
const DIM_ONLY: &str = "This is a netCDF dimension but not a netCDF variable.";
/// Attributes netCDF-4 uses for its own bookkeeping; never shown as user attributes.
const HIDDEN: &[&str] = &[
    "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_NCProperties",
    "_nc3_strict",
];

fn bad(message: impl Into<String>) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: message.into(),
    }
}

// ----- HDF5 values as annoray values -----

fn dtype_of(t: ValueType) -> DType {
    match t {
        ValueType::I8 => DType::I8,
        ValueType::U8 => DType::U8,
        ValueType::I16 => DType::I16,
        ValueType::U16 => DType::U16,
        ValueType::I32 => DType::I32,
        ValueType::U32 => DType::U32,
        ValueType::I64 => DType::I64,
        ValueType::U64 => DType::U64,
        ValueType::F32 => DType::F32,
        ValueType::F64 => DType::F64,
        _ => DType::Str,
    }
}

fn value_type(d: DType) -> Option<ValueType> {
    Some(match d {
        DType::I8 => ValueType::I8,
        DType::U8 => ValueType::U8,
        DType::I16 => ValueType::I16,
        DType::U16 => ValueType::U16,
        DType::I32 => ValueType::I32,
        DType::U32 => ValueType::U32,
        DType::I64 => ValueType::I64,
        DType::U64 => ValueType::U64,
        DType::F32 => ValueType::F32,
        DType::F64 => ValueType::F64,
        DType::Str => ValueType::Str,
        _ => return None,
    })
}

/// HDF5 values as a typed array (no copy).
fn typed(a: Array) -> Result<TypedArray<Owned>> {
    Ok(match a {
        Array::I8(x) => TypedArray::I8(x),
        Array::U8(x) => TypedArray::U8(x),
        Array::I16(x) => TypedArray::I16(x),
        Array::U16(x) => TypedArray::U16(x),
        Array::I32(x) => TypedArray::I32(x),
        Array::U32(x) => TypedArray::U32(x),
        Array::I64(x) => TypedArray::I64(x),
        Array::U64(x) => TypedArray::U64(x),
        Array::F32(x) => TypedArray::F32(x),
        Array::F64(x) => TypedArray::F64(x),
        Array::Str(x) => TypedArray::Str(x),
        other => {
            return Err(IoError::Unsupported(format!(
                "HDF5 values of type {}",
                other.dtype()
            )));
        }
    })
}

/// A typed array as HDF5 values (no copy).
pub(crate) fn hdf5_array(a: TypedArray<Owned>) -> Result<Array> {
    Ok(match a {
        TypedArray::I8(x) => Array::I8(x),
        TypedArray::U8(x) => Array::U8(x),
        TypedArray::I16(x) => Array::I16(x),
        TypedArray::U16(x) => Array::U16(x),
        TypedArray::I32(x) => Array::I32(x),
        TypedArray::U32(x) => Array::U32(x),
        TypedArray::I64(x) => Array::I64(x),
        TypedArray::U64(x) => Array::U64(x),
        TypedArray::F32(x) => Array::F32(x),
        TypedArray::F64(x) => Array::F64(x),
        TypedArray::Str(x) => Array::Str(x),
        // Complex variables are compounds, written with the whole dataset.
        TypedArray::C64(_) | TypedArray::C128(_) => {
            return Err(IoError::Unsupported(
                "complex values are written with the whole dataset, not into an existing file"
                    .into(),
            ));
        }
        other => {
            return Err(IoError::Unsupported(format!(
                "element type {} must be CF-encoded before writing",
                other.dtype()
            )));
        }
    })
}

/// An attribute value in the netCDF attribute model (`None` if it cannot be represented).
fn attr_value(a: &annoray_hdf5::Attribute) -> Option<AttrValue> {
    let v = a.value().ok()?;
    let scalar = a.shape.as_ref().is_some_and(Vec::is_empty);
    Some(match v {
        Array::Str(s) if scalar => AttrValue::Str(s.iter().next().cloned().unwrap_or_default()),
        Array::Str(s) => AttrValue::Strs(s.into_iter().collect()),
        Array::I8(x) => AttrValue::I8(x.into_iter().collect()),
        Array::I16(x) => AttrValue::I16(x.into_iter().collect()),
        Array::I32(x) => AttrValue::I32(x.into_iter().collect()),
        Array::I64(x) => AttrValue::I64(x.into_iter().collect()),
        Array::U8(x) => AttrValue::U8(x.into_iter().collect()),
        Array::U16(x) => AttrValue::U16(x.into_iter().collect()),
        Array::U32(x) => AttrValue::U32(x.into_iter().collect()),
        Array::U64(x) => AttrValue::U64(x.into_iter().collect()),
        Array::F32(x) => AttrValue::F32(x.into_iter().collect()),
        Array::F64(x) => AttrValue::F64(x.into_iter().collect()),
        _ => return None,
    })
}

/// An attribute to write: text as an exact-length fixed string, everything else 1-D.
fn attr_data(v: &AttrValue) -> AttrData {
    fn one<T>(v: &[T]) -> ArrayD<T>
    where
        T: Clone,
    {
        ndarray::Array1::from(v.to_vec()).into_dyn()
    }
    AttrData::Array(match v {
        AttrValue::Str(s) => return AttrData::Text(s.clone()),
        AttrValue::Strs(x) => Array::Str(one(x)),
        AttrValue::I8(x) => Array::I8(one(x)),
        AttrValue::I16(x) => Array::I16(one(x)),
        AttrValue::I32(x) => Array::I32(one(x)),
        AttrValue::I64(x) => Array::I64(one(x)),
        AttrValue::U8(x) => Array::U8(one(x)),
        AttrValue::U16(x) => Array::U16(one(x)),
        AttrValue::U32(x) => Array::U32(one(x)),
        AttrValue::U64(x) => Array::U64(one(x)),
        AttrValue::F32(x) => Array::F32(one(x)),
        AttrValue::F64(x) => Array::F64(one(x)),
    })
}

/// A lazy variable's values, read slice by slice as they are written.
#[derive(Debug)]
struct Streamed {
    source: Arc<dyn LazySource>,
    dtype: ValueType,
}

/// Values borrowed from the dataset being written, served slice by slice.
#[derive(Debug)]
struct Viewed<'a> {
    data: TypedArray<crate::View<'a>>,
    dtype: ValueType,
}

impl annoray_hdf5::Source for Viewed<'_> {
    fn dtype(&self) -> ValueType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn read_slice(
        &self,
        start: &[usize],
        count: &[usize],
    ) -> std::result::Result<Array, Box<dyn std::error::Error + Send + Sync>> {
        Ok(hdf5_array(super::raw::slice_of(&self.data, start, count))?)
    }
}

impl annoray_hdf5::Source for Streamed {
    fn dtype(&self) -> ValueType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        self.source.shape()
    }

    fn read_slice(
        &self,
        start: &[usize],
        count: &[usize],
    ) -> std::result::Result<Array, Box<dyn std::error::Error + Send + Sync>> {
        Ok(hdf5_array(self.source.read_slice(start, count)?)?)
    }
}

// ----- reading -----

/// A dimension scale found while reading.
struct Scale {
    name: Name,
    id: Option<i32>,
    dim: RawDim,
    /// A dimension with no variable of its own.
    dim_only: bool,
}

fn string_attr(d: &Dataset, name: &str) -> Result<Option<String>> {
    Ok(d.attrs()?
        .into_iter()
        .find(|a| a.name == name)
        .and_then(|a| match attr_value(&a) {
            Some(AttrValue::Str(s)) => Some(s),
            _ => None,
        }))
}

fn int_attr(d: &Dataset, name: &str) -> Result<Option<i32>> {
    Ok(d.attrs()?
        .into_iter()
        .find(|a| a.name == name)
        .and_then(|a| match attr_value(&a) {
            Some(AttrValue::I32(v)) => v.first().copied(),
            _ => None,
        }))
}

fn scale_of(name: &str, d: &Dataset) -> Result<Option<Scale>> {
    if string_attr(d, "CLASS")?.as_deref() != Some("DIMENSION_SCALE") {
        return Ok(None);
    }
    let dim_only = string_attr(d, "NAME")?.is_some_and(|n| n.starts_with(DIM_ONLY));
    let len = d.shape().first().copied().unwrap_or(0);
    let unlimited = d.max_shape().first().is_some_and(Option::is_none);
    let name = name.strip_prefix(NON_COORD).unwrap_or(name);
    Ok(Some(Scale {
        name: Name::from(name),
        id: int_attr(d, "_Netcdf4Dimid")?,
        dim: RawDim { len, unlimited },
        dim_only,
    }))
}

/// Dimension names visible from a group: by scale address and by dimension id.
#[derive(Clone, Default)]
struct Scope {
    by_addr: HashMap<u64, Name>,
    by_id: HashMap<i32, Name>,
}

fn user_attrs(
    attrs: Vec<annoray_hdf5::Attribute>,
    skipped: &mut Vec<(Name, String)>,
    owner: &str,
) -> Attrs {
    let mut out = Attrs::new();
    for a in attrs {
        if HIDDEN.contains(&a.name.as_str()) {
            continue;
        }
        match attr_value(&a) {
            Some(v) => {
                out.insert(Name::from(a.name.as_str()), v);
            }
            None => skipped.push((
                Name::from(format!("{owner}@{}", a.name)),
                "unsupported attribute type".into(),
            )),
        }
    }
    out
}

/// `NC_CHAR` data: HDF5 1-byte strings back to bytes.
fn chars(data: TypedArray<Owned>) -> TypedArray<Owned> {
    match data {
        TypedArray::Str(a) => {
            TypedArray::U8(a.mapv(|s| s.as_bytes().first().copied().unwrap_or(0)))
        }
        other => other,
    }
}

/// One variable of an open file, read slice by slice.
#[derive(Debug)]
struct H5Source {
    dataset: Dataset,
    dtype: DType,
    shape: Vec<usize>,
    is_char: bool,
    is_bool: bool,
}

impl LazySource for H5Source {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let size = *self.dataset.chunks()?.get(axis)?;
        (size > 0).then_some((size, 0))
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> crate::Result<TypedArray<Owned>> {
        let data = self
            .dataset
            .read_slice(start, count)
            .map_err(|e| CoreError::Read(e.to_string()))
            .and_then(|a| typed(a).map_err(|e| CoreError::Read(e.to_string())))?;
        Ok(if self.is_char {
            chars(data)
        } else if self.is_bool {
            bools(&data)
        } else {
            data
        })
    }
}

/// h5py's booleans: an enumeration of `FALSE = 0` and `TRUE = 1` over a 1-byte integer.
fn is_h5py_bool(t: &Datatype) -> bool {
    match t {
        Datatype::Enum { base, members } => {
            base.size() == 1
                && members.len() == 2
                && members
                    .iter()
                    .any(|(n, v)| n == "FALSE" && v.as_slice() == [0])
                && members
                    .iter()
                    .any(|(n, v)| n == "TRUE" && v.as_slice() == [1])
        }
        _ => false,
    }
}

/// 1-byte integers as booleans (non-zero is true, as h5py reads them).
fn bools(data: &TypedArray<Owned>) -> TypedArray<Owned> {
    match data {
        TypedArray::I8(a) => TypedArray::Bool(a.mapv(|x| x != 0)),
        TypedArray::U8(a) => TypedArray::Bool(a.mapv(|x| x != 0)),
        other => other.clone(),
    }
}

/// An enumeration's members, their values in the base integer type; named after the
/// committed datatype the dataset's type is, else one equal to it, as netCDF-C matches types
/// (`named`: the named datatypes in scope).
fn enum_type(d: &Dataset, named: &[(String, u64, Datatype)]) -> Option<super::raw::EnumType> {
    let Datatype::Enum { base, members } = d.dtype() else {
        return None;
    };
    let Datatype::Int {
        signed, big_endian, ..
    } = base.as_ref()
    else {
        return None;
    };
    let value = |b: &[u8]| -> Option<i128> {
        let mut le = b.to_vec();
        if *big_endian {
            le.reverse();
        }
        if le.len() > 8 {
            return None;
        }
        let negative = *signed && le.last().is_some_and(|x| x & 0x80 != 0);
        let mut buf = [if negative { 0xFF } else { 0 }; 16];
        buf.get_mut(..le.len())?.copy_from_slice(&le);
        Some(i128::from_le_bytes(buf))
    };
    let members = members
        .iter()
        .map(|(n, b)| Some((n.clone(), value(b)?)))
        .collect::<Option<Vec<_>>>()?;
    let name = d
        .committed_type()
        .and_then(|at| named.iter().find(|(_, a, _)| *a == at))
        .or_else(|| named.iter().find(|(_, _, t)| t == d.dtype()))
        .map(|(n, _, _)| n.clone());
    Some(super::raw::EnumType { name, members })
}

/// A name not yet used by a variable or dimension of `ds`: `base`, else `base_2`, `base_3`, ….
fn fresh(ds: &RawDataset<'_>, base: &str) -> Name {
    let taken = |n: &str| ds.vars.contains_key(n) || ds.dims.contains_key(n);
    if !taken(base) {
        return Name::from(base);
    }
    (2..)
        .map(|k| format!("{base}_{k}"))
        .find(|n| !taken(n))
        .map_or_else(|| Name::from(base), |n| Name::from(n.as_str()))
}

/// The member names of a complex number stored as a compound: h5py's and nc-complex's
/// `r`/`i`, and `real`/`imag`.
const COMPLEX_NAMES: [(&str, &str); 2] = [("r", "i"), ("real", "imag")];

/// The size of each part (4 or 8 bytes) when `t` is a complex number: a compound of exactly
/// two floats of one size and byte order, real part first, named as [`COMPLEX_NAMES`].
fn complex_part_size(t: &Datatype) -> Option<usize> {
    let Datatype::Compound { size, members } = t else {
        return None;
    };
    let [re, im] = members.as_slice() else {
        return None;
    };
    let (
        Datatype::Float {
            size: a,
            big_endian: ea,
        },
        Datatype::Float {
            size: b,
            big_endian: eb,
        },
    ) = (&re.dtype, &im.dtype)
    else {
        return None;
    };
    let part = usize::from(*a);
    let named = COMPLEX_NAMES
        .iter()
        .any(|(r, i)| re.name.eq_ignore_ascii_case(r) && im.name.eq_ignore_ascii_case(i));
    (named
        && a == b
        && ea == eb
        && matches!(part, 4 | 8)
        && re.offset == 0
        && im.offset == part
        && *size == part.saturating_mul(2))
    .then_some(part)
}

/// A complex compound dataset (see [`complex_part_size`]) as `complex64` or `complex128`.
fn read_complex(d: &Dataset, part: usize) -> Result<TypedArray<Owned>> {
    let fields = d.read_compound()?;
    let mut fields = fields.into_iter();
    let (Some((_, re)), Some((_, im))) = (fields.next(), fields.next()) else {
        return Err(bad("complex compound without two parts"));
    };
    let shape = IxDyn(&d.shape());
    let shaped = |e: ndarray::ShapeError| bad(format!("complex compound: {e}"));
    Ok(match (part, re, im) {
        (4, Array::F32(re), Array::F32(im)) => {
            let v = re
                .iter()
                .zip(im.iter())
                .map(|(r, i)| num_complex::Complex::new(*r, *i))
                .collect();
            TypedArray::C64(ArrayD::from_shape_vec(shape, v).map_err(shaped)?)
        }
        (8, Array::F64(re), Array::F64(im)) => {
            let v = re
                .iter()
                .zip(im.iter())
                .map(|(r, i)| num_complex::Complex::new(*r, *i))
                .collect();
            TypedArray::C128(ArrayD::from_shape_vec(shape, v).map_err(shaped)?)
        }
        _ => return Err(bad("complex compound parts of unexpected types")),
    })
}

/// A complex variable as an HDF5 compound of two floats `r` and `i` (h5py's layout, which
/// netCDF4-python and nc-complex read as complex numbers).
fn complex_compound(v: &RawVariable<'_>) -> Result<Option<annoray_hdf5::WCompound>> {
    let values = v.data.values()?;
    let big_endian = v.storage.big_endian;
    let compound = |part: u8, re: Array, im: Array| {
        let float = Datatype::Float {
            size: part,
            big_endian,
        };
        let size = usize::from(part);
        annoray_hdf5::WCompound {
            dtype: Datatype::Compound {
                size: size.saturating_mul(2),
                members: vec![
                    annoray_hdf5::Member {
                        name: "r".into(),
                        offset: 0,
                        dtype: float.clone(),
                    },
                    annoray_hdf5::Member {
                        name: "i".into(),
                        offset: size,
                        dtype: float,
                    },
                ],
            },
            shape: values.shape().to_vec(),
            fields: vec![("r".into(), re), ("i".into(), im)],
        }
    };
    Ok(match values.as_ref() {
        TypedArray::C64(a) => Some(compound(
            4,
            Array::F32(a.mapv(|z| z.re)),
            Array::F32(a.mapv(|z| z.im)),
        )),
        TypedArray::C128(a) => Some(compound(
            8,
            Array::F64(a.mapv(|z| z.re)),
            Array::F64(a.mapv(|z| z.im)),
        )),
        _ => None,
    })
}

/// A compound (record) variable as one variable per leaf field, `name.field` (dotted for
/// nested members), on the variable's dimensions plus `name.field_dimK` for array members.
/// Every field carries the variable's attributes, and in its storage the record type.
fn compound(
    d: &Dataset,
    name: &Name,
    dims: &[Name],
    attrs: &Attrs,
    ds: &mut RawDataset<'static>,
) -> Result<()> {
    let fields = d.read_compound()?;
    let dtype = Arc::new(d.dtype().clone());
    // All or nothing: names are checked before anything is added.
    let named: Vec<(Name, String, TypedArray<Owned>)> = fields
        .into_iter()
        .map(|(path, values)| Ok((fresh(ds, &format!("{name}.{path}")), path, typed(values)?)))
        .collect::<Result<_>>()?;
    for (var, path, values) in named {
        let mut vdims = dims.to_vec();
        for k in dims.len()..values.ndim() {
            let dim = fresh(ds, &format!("{var}_dim{}", k.saturating_sub(dims.len())));
            ds.dims.insert(
                dim.clone(),
                RawDim {
                    len: values.shape().get(k).copied().unwrap_or(0),
                    unlimited: false,
                },
            );
            vdims.push(dim);
        }
        ds.vars.insert(
            var,
            RawVariable {
                dims: vdims,
                data: values.into(),
                is_char: false,
                attrs: attrs.clone(),
                storage: Storage {
                    compound: Some(super::raw::CompoundField {
                        parent: name.clone(),
                        path,
                        dtype: Arc::clone(&dtype),
                    }),
                    ..Storage::default()
                },
            },
        );
    }
    Ok(())
}

/// A variable-length (ragged) variable as a CF contiguous ragged array: `name(name_obs)` holds
/// every element's values in order, `name_count(dims…)` how many belong to each element (its
/// `sample_dimension` attribute names `name_obs`).
fn ragged(
    d: &Dataset,
    name: &Name,
    dims: Vec<Name>,
    attrs: Attrs,
    ds: &mut RawDataset<'static>,
) -> Result<()> {
    let (counts, values) = d.read_ragged()?;
    let values = typed(values)?;
    let total = values.len();
    let obs = fresh(ds, &format!("{name}_obs"));
    let count_name = fresh(ds, &format!("{name}_count"));
    let biggest = counts.iter().copied().max().unwrap_or(0);
    let counts: TypedArray<Owned> = if i32::try_from(biggest).is_ok() {
        TypedArray::I32(counts.mapv(|c| i32::try_from(c).unwrap_or(i32::MAX)))
    } else {
        TypedArray::I64(counts.mapv(|c| i64::try_from(c).unwrap_or(i64::MAX)))
    };
    ds.dims.insert(
        obs.clone(),
        RawDim {
            len: total,
            unlimited: false,
        },
    );
    let (deflate, shuffle) = d.compression();
    ds.vars.insert(
        name.clone(),
        RawVariable {
            dims: vec![obs.clone()],
            data: values.into(),
            is_char: false,
            attrs,
            storage: Storage {
                deflate,
                shuffle,
                ragged: Some(count_name.clone()),
                ..Storage::default()
            },
        },
    );
    let mut count_attrs = Attrs::new();
    count_attrs.insert(
        Name::from("sample_dimension"),
        AttrValue::Str(obs.to_string()),
    );
    ds.vars.insert(
        count_name,
        RawVariable {
            dims,
            data: counts.into(),
            is_char: false,
            attrs: count_attrs,
            storage: Storage::default(),
        },
    );
    Ok(())
}

fn read_group(g: &Group, parent: &Scope, lazy: bool) -> Result<RawDataset<'static>> {
    let mut ds = RawDataset::default();
    let mut scope = parent.clone();
    let (members, skipped_links) = g.members_and_skipped()?;
    for (name, why) in skipped_links {
        ds.skipped.push((Name::from(name.as_str()), why));
    }

    // Dimensions: the scales of this group, ordered by dimension id.
    let mut scales: Vec<(u64, Scale)> = Vec::new();
    for (name, node) in &members {
        if let Node::Dataset(d) = node
            && let Some(s) = scale_of(name, d)?
        {
            scales.push((d.address(), s));
        }
    }
    scales.sort_by_key(|(_, s)| s.id.unwrap_or(i32::MAX));
    for (addr, s) in &scales {
        scope.by_addr.insert(*addr, s.name.clone());
        if let Some(id) = s.id {
            scope.by_id.insert(id, s.name.clone());
        }
        ds.dims.insert(s.name.clone(), s.dim);
    }

    // Root (or group) attributes.
    ds.attrs = user_attrs(g.attrs()?, &mut ds.skipped, g.path());

    let mut type_names: Option<Vec<(String, u64, Datatype)>> = None;
    for (name, node) in members {
        match node {
            Node::Group(child) => {
                ds.groups
                    .insert(Name::from(name.as_str()), read_group(&child, &scope, lazy)?);
            }
            Node::Dataset(d) => {
                if scales.iter().any(|(a, s)| *a == d.address() && s.dim_only) {
                    continue;
                }
                let var_name = Name::from(name.strip_prefix(NON_COORD).unwrap_or(&name));
                let attrs = d.attrs()?;
                let shape = d.shape();
                // Dimensions: DIMENSION_LIST, else the dataset is its own scale, else
                // _Netcdf4Coordinates, else phony dimensions.
                let dim_list = attrs
                    .iter()
                    .find(|a| a.name == "DIMENSION_LIST")
                    .map(annoray_hdf5::Attribute::reference_lists)
                    .transpose()?;
                let mut dims: Vec<Name> = Vec::new();
                if let Some(lists) = dim_list {
                    for (axis, refs) in lists.iter().enumerate() {
                        let n = refs
                            .iter()
                            .find_map(|r| scope.by_addr.get(r))
                            .cloned()
                            .ok_or_else(|| {
                                bad(format!(
                                    "{}: dimension {axis} refers to an unknown scale",
                                    d.path()
                                ))
                            })?;
                        dims.push(n);
                    }
                } else if shape.len() == 1 && scales.iter().any(|(a, _)| *a == d.address()) {
                    dims.push(var_name.clone());
                } else if let Some(AttrValue::I32(ids)) = attrs
                    .iter()
                    .find(|a| a.name == "_Netcdf4Coordinates")
                    .and_then(attr_value)
                {
                    for id in ids {
                        dims.push(scope.by_id.get(&id).cloned().ok_or_else(|| {
                            bad(format!("{}: unknown dimension id {id}", d.path()))
                        })?);
                    }
                }
                if dims.len() != shape.len() {
                    // Plain HDF5: phony dimensions, reused by length within the group.
                    dims.clear();
                    for len in &shape {
                        let existing = ds
                            .dims
                            .iter()
                            .find(|(n, x)| {
                                x.len == *len && n.starts_with("phony_dim_") && !dims.contains(n)
                            })
                            .map(|(n, _)| n.clone());
                        let n = existing.unwrap_or_else(|| {
                            let n = Name::from(format!("phony_dim_{}", ds.dims.len()));
                            ds.dims.insert(
                                n.clone(),
                                RawDim {
                                    len: *len,
                                    unlimited: false,
                                },
                            );
                            n
                        });
                        dims.push(n);
                    }
                }
                if let Some(part) = complex_part_size(d.dtype()) {
                    let data = match read_complex(&d, part) {
                        Ok(t) => t,
                        Err(e) => {
                            ds.skipped.push((var_name, e.to_string()));
                            continue;
                        }
                    };
                    let (deflate, shuffle) = d.compression();
                    let storage = Storage {
                        chunks: d.chunks(),
                        deflate,
                        shuffle,
                        compression: d.codec(),
                        fletcher32: d.fletcher32(),
                        contiguous: d.contiguous(),
                        ..Storage::default()
                    };
                    let attrs = user_attrs(attrs, &mut ds.skipped, d.path());
                    ds.vars.insert(
                        var_name,
                        RawVariable {
                            dims,
                            data: data.into(),
                            is_char: false,
                            attrs,
                            storage,
                        },
                    );
                    continue;
                }
                if matches!(d.dtype(), Datatype::Compound { .. }) {
                    let user = user_attrs(attrs, &mut ds.skipped, d.path());
                    if let Err(e) = compound(&d, &var_name, &dims, &user, &mut ds) {
                        ds.skipped.push((var_name, e.to_string()));
                    }
                    continue;
                }
                if matches!(d.dtype(), Datatype::VarSeq { .. }) {
                    let user = user_attrs(attrs, &mut ds.skipped, d.path());
                    if let Err(e) = ragged(&d, &var_name, dims, user, &mut ds) {
                        ds.skipped.push((var_name, e.to_string()));
                    }
                    continue;
                }
                let is_char = matches!(d.dtype(), Datatype::FixedString { size: 1, .. });
                let is_bool = is_h5py_bool(d.dtype());
                let value_dtype = d.dtype().value_dtype().map(dtype_of);
                let data = match (lazy, value_dtype) {
                    (true, Some(dtype)) => Ok(RawData::Lazy(Arc::new(H5Source {
                        dataset: Dataset::clone(&d),
                        dtype: if is_char {
                            DType::U8
                        } else if is_bool {
                            DType::Bool
                        } else {
                            dtype
                        },
                        shape: shape.clone(),
                        is_char,
                        is_bool,
                    }))),
                    _ => d.read().map_err(IoError::from).and_then(typed).map(|v| {
                        if is_char {
                            chars(v)
                        } else if is_bool {
                            bools(&v)
                        } else {
                            v
                        }
                        .into()
                    }),
                };
                let enumeration = if is_bool {
                    None
                } else if matches!(d.dtype(), Datatype::Enum { .. }) {
                    if type_names.is_none() {
                        // netCDF-C looks for a type in the variable's group, then its parents;
                        // the group and the root cover what it writes.
                        let mut names = g.datatypes().unwrap_or_default();
                        if g.path() != "/"
                            && let Ok(root) = g.file().root()
                        {
                            names.extend(root.datatypes().unwrap_or_default());
                        }
                        type_names = Some(names);
                    }
                    enum_type(&d, type_names.as_deref().unwrap_or_default())
                } else {
                    None
                };
                let data = match data {
                    Ok(v) => v,
                    Err(e) => {
                        ds.skipped.push((var_name, e.to_string()));
                        continue;
                    }
                };
                let (deflate, shuffle) = d.compression();
                let storage = Storage {
                    chunks: d.chunks(),
                    deflate,
                    shuffle,
                    compression: d.codec(),
                    fletcher32: d.fletcher32(),
                    big_endian: d.big_endian(),
                    contiguous: d.contiguous(),
                    float16: d.half(),
                    ragged: None,
                    compound: None,
                    enumeration,
                };
                let attrs = user_attrs(attrs, &mut ds.skipped, d.path());
                ds.vars.insert(
                    var_name,
                    RawVariable {
                        dims,
                        data,
                        is_char,
                        attrs,
                        storage,
                    },
                );
            }
            _ => {}
        }
    }
    Ok(ds)
}

/// Read a netCDF-4 (or plain HDF5) file held in memory.
///
/// # Errors
/// [`IoError::Backend`] for unreadable files.
pub fn read_bytes(bytes: Vec<u8>) -> Result<RawDataset<'static>> {
    read_file_bytes(annoray_hdf5::FileBytes::Owned(bytes), false)
}

/// Read a netCDF-4 (or plain HDF5) file from its bytes; with `lazy`, values are read from
/// them on demand.
///
/// # Errors
/// [`IoError::Backend`] for unreadable files.
pub fn read_file_bytes(bytes: annoray_hdf5::FileBytes, lazy: bool) -> Result<RawDataset<'static>> {
    read_group(
        &File::from_file_bytes(bytes)?.root()?,
        &Scope::default(),
        lazy,
    )
}

// ----- writing -----

/// Raw little-endian bytes of the first element of a numeric attribute, in element type `d`.
fn fill_bytes(a: &AttrValue, d: DType) -> Option<Vec<u8>> {
    Some(match (a, d) {
        (AttrValue::I8(v), DType::I8) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::U8(v), DType::U8) => vec![*v.first()?],
        (AttrValue::I16(v), DType::I16) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::U16(v), DType::U16) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::I32(v), DType::I32) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::U32(v), DType::U32) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::I64(v), DType::I64) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::U64(v), DType::U64) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::F32(v), DType::F32) => v.first()?.to_le_bytes().to_vec(),
        (AttrValue::F64(v), DType::F64) => v.first()?.to_le_bytes().to_vec(),
        _ => return None,
    })
}

/// netCDF-C's default chunk cache size, which its default chunk sizes aim for.
const DEFAULT_CHUNK_SIZE: f64 = 16_777_216.0;
const DEFAULT_1D_UNLIM_SIZE: usize = 4096;

/// netCDF-C's default chunk shape (`nc4_find_default_chunksizes2`), ported exactly so files we
/// write perform like netCDF-C's. Unlimited dimensions have length 0 at definition time in
/// netCDF-C, so they never get the overhang adjustment.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn default_chunks(shape: &[usize], unlimited: &[bool], type_size: usize) -> Vec<usize> {
    let ndims = shape.len();
    let ts = type_size.max(1);
    let mut chunks = vec![0usize; ndims];
    let mut num_values: f32 = 1.0;
    let mut num_unlim = 0usize;
    for ((c, len), u) in chunks.iter_mut().zip(shape).zip(unlimited) {
        if *u {
            num_unlim = num_unlim.saturating_add(1);
            *c = 1;
        } else {
            num_values *= *len as f32;
        }
    }
    let big = 16_777_216usize;
    if ndims == 1 && num_unlim == 1 {
        let per = big.checked_div(ts).unwrap_or(1);
        let suggested = if per == 0 {
            1
        } else {
            per.min(DEFAULT_1D_UNLIM_SIZE)
        };
        if let Some(c) = chunks.first_mut() {
            *c = suggested.checked_div(ts).unwrap_or(1);
        }
    }
    if ndims > 1 && num_unlim == ndims {
        let s = (DEFAULT_CHUNK_SIZE / ts as f64).powf(1.0 / ndims as f64) as usize;
        chunks.iter_mut().for_each(|c| *c = s.max(1));
    }
    let fixed = ndims.saturating_sub(num_unlim) as f64;
    for (c, len) in chunks.iter_mut().zip(shape) {
        if *c == 0 {
            let s = ((DEFAULT_CHUNK_SIZE / (f64::from(num_values) * ts as f64)).powf(1.0 / fixed)
                * *len as f64
                - 0.5) as usize;
            *c = s.min(*len).max(1);
        }
    }
    // Too big for HDF5's 32-bit chunk size? Halve every dimension until it fits.
    let too_big =
        |c: &[usize]| c.iter().fold(ts as f64, |a, x| a * *x as f64) > f64::from(u32::MAX);
    while too_big(&chunks) {
        chunks.iter_mut().for_each(|c| *c = (*c / 2).max(1));
    }
    // Spread any overhang evenly over the chunks.
    for ((c, len), u) in chunks.iter_mut().zip(shape).zip(unlimited) {
        let len = if *u { 0 } else { *len };
        let n = len
            .saturating_add(c.saturating_sub(1))
            .checked_div(*c)
            .unwrap_or(0);
        if n > 0 {
            let overhang = n.saturating_mul(*c).saturating_sub(len);
            *c = c.saturating_sub(overhang.checked_div(n).unwrap_or(0));
        }
    }
    chunks
}

fn element_size(d: DType) -> usize {
    match d {
        DType::I8 | DType::U8 => 1,
        DType::I16 | DType::U16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        _ => 8, // 64-bit numbers, and strings (a pointer in netCDF-C's memory model)
    }
}

fn path_join(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

/// Where each dimension's scale lives, and its id.
#[derive(Clone, Default)]
struct WScope {
    dims: HashMap<Name, (String, i32, RawDim)>,
}

/// The group with each CF contiguous ragged array read from a variable-length variable (its
/// values' storage names the count variable) merged back into one variable over the count
/// variable's dimensions, with each element's length; `None` when there is none.
/// Variables to write in a form of their own: ragged arrays as variable-length variables
/// (each element's length), compound fields as one compound variable.
#[derive(Default)]
struct Merged {
    counts: HashMap<Name, ArrayD<u64>>,
    compounds: HashMap<Name, annoray_hdf5::WCompound>,
}

/// The group with ragged arrays and compound fields merged back (see [`merge_ragged`],
/// [`merge_compounds`]); `None` when there is nothing to merge.
fn merge<'a>(ds: &RawDataset<'a>) -> Result<Option<(RawDataset<'a>, Merged)>> {
    let ragged = ds.vars.values().any(|v| v.storage.ragged.is_some());
    let compound = ds.vars.values().any(|v| v.storage.compound.is_some());
    let complex = ds
        .vars
        .values()
        .any(|v| matches!(v.data.dtype(), DType::C64 | DType::C128));
    if !ragged && !compound && !complex {
        return Ok(None);
    }
    let mut out = ds.clone();
    let mut merged = Merged::default();
    if ragged {
        merged.counts = merge_ragged(&mut out)?;
    }
    if compound {
        merged.compounds = merge_compounds(&mut out)?;
    }
    if complex {
        for (name, v) in &out.vars {
            if let Some(c) = complex_compound(v)? {
                merged.compounds.insert(name.clone(), c);
            }
        }
    }
    Ok(Some((out, merged)))
}

/// Fields of compound variables (each field's storage names its variable, member and the
/// record type) packed back into one compound variable per variable, when every member is
/// there on the same dimensions. The values are converted to the members' types (checked:
/// a value a member cannot hold is an error).
fn merge_compounds(out: &mut RawDataset<'_>) -> Result<HashMap<Name, annoray_hdf5::WCompound>> {
    let mut parents: Vec<Name> = Vec::new();
    for v in out.vars.values() {
        if let Some(c) = &v.storage.compound
            && !parents.contains(&c.parent)
        {
            parents.push(c.parent.clone());
        }
    }
    let mut packed = HashMap::new();
    for parent in parents {
        let fields: Vec<(Name, super::raw::CompoundField)> = out
            .vars
            .iter()
            .filter_map(|(n, v)| {
                v.storage
                    .compound
                    .as_ref()
                    .filter(|c| c.parent == parent)
                    .map(|c| (n.clone(), c.clone()))
            })
            .collect();
        let Some((_, first)) = fields.first() else {
            continue;
        };
        let Datatype::Compound { members, .. } = first.dtype.as_ref() else {
            continue;
        };
        let leaves = annoray_hdf5::compound_fields(members)?;
        // Every member, from the same record type, or the fields stay separate variables.
        let complete = leaves.len() == fields.len()
            && leaves.iter().all(|(path, _, _)| {
                fields
                    .iter()
                    .any(|(_, f)| f.path == *path && f.dtype == first.dtype)
            });
        if !complete || out.vars.contains_key(&parent) {
            continue;
        }
        let mut base_dims: Option<Vec<Name>> = None;
        let mut columns = Vec::with_capacity(leaves.len());
        let mut extra_dims = Vec::new();
        let mut attrs: Option<Attrs> = None;
        for (path, member, member_dims) in &leaves {
            let (name, _) = fields
                .iter()
                .find(|(_, f)| f.path == *path)
                .ok_or_else(|| bad("compound field"))?;
            let v = out.vars.get(name).ok_or_else(|| bad("compound field"))?;
            let keep = v.dims.len().saturating_sub(member_dims.len());
            let (own, extra) = v.dims.split_at(keep.min(v.dims.len()));
            match &base_dims {
                Some(b) if b.as_slice() != own => {
                    return Err(bad(format!(
                        "compound {parent:?}: field {name:?} has dimensions {own:?}, not {b:?}"
                    )));
                }
                Some(_) => {}
                None => base_dims = Some(own.to_vec()),
            }
            extra_dims.extend(extra.iter().cloned());
            let values = v.data.values()?.into_owned();
            let values = match member.value_dtype() {
                Some(ValueType::Str) => values,
                Some(t) => {
                    let to = dtype_of(t);
                    if values.dtype() == to {
                        values
                    } else {
                        crate::cast::cast(&values.view(), to)
                            .map_err(|e| bad(format!("compound {parent:?}, field {path:?}: {e}")))?
                    }
                }
                None => {
                    return Err(IoError::Unsupported(format!(
                        "compound {parent:?}: writing field {path:?}"
                    )));
                }
            };
            columns.push((path.clone(), hdf5_array(values)?));
            // Attributes all fields share (the variable's, copied to each when read).
            let mine: Attrs = v
                .attrs
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "_FillValue" | "missing_value"))
                .map(|(k, a)| (k.clone(), a.clone()))
                .collect();
            attrs = Some(match attrs {
                None => mine,
                Some(common) => common
                    .into_iter()
                    .filter(|(k, a)| mine.get(k) == Some(a))
                    .collect(),
            });
        }
        let base_dims = base_dims.unwrap_or_default();
        let shape: Vec<usize> = base_dims
            .iter()
            .map(|d| out.dims.get(d).map_or(0, |x| x.len))
            .collect();
        for (name, _) in &fields {
            out.vars.shift_remove(name);
        }
        for d in extra_dims {
            if !out.vars.values().any(|x| x.dims.contains(&d)) {
                out.dims.shift_remove(&d);
            }
        }
        out.vars.insert(
            parent.clone(),
            RawVariable {
                dims: base_dims,
                data: TypedArray::<Owned>::I8(ArrayD::zeros(IxDyn(&shape))).into(),
                is_char: false,
                attrs: attrs.unwrap_or_default(),
                storage: Storage::default(),
            },
        );
        packed.insert(
            parent,
            annoray_hdf5::WCompound {
                dtype: first.dtype.as_ref().clone(),
                shape,
                fields: columns,
            },
        );
    }
    Ok(packed)
}

/// The HDF5 enumeration a variable's [`EnumType`](super::raw::EnumType) describes, over the
/// variable's integer type (in its byte order); every value must fit that type.
fn enum_datatype(name: &Name, v: &RawVariable<'_>, e: &super::raw::EnumType) -> Result<Datatype> {
    let dtype = v.data.dtype();
    let (size, signed): (u8, bool) = match dtype {
        DType::I8 => (1, true),
        DType::U8 => (1, false),
        DType::I16 => (2, true),
        DType::U16 => (2, false),
        DType::I32 => (4, true),
        DType::U32 => (4, false),
        DType::I64 => (8, true),
        DType::U64 => (8, false),
        t => {
            return Err(bad(format!(
                "{name:?}: an enumeration needs integer values, not {t}"
            )));
        }
    };
    let fits = |x: i128| match dtype {
        DType::I8 => i8::try_from(x).is_ok(),
        DType::U8 => u8::try_from(x).is_ok(),
        DType::I16 => i16::try_from(x).is_ok(),
        DType::U16 => u16::try_from(x).is_ok(),
        DType::I32 => i32::try_from(x).is_ok(),
        DType::U32 => u32::try_from(x).is_ok(),
        DType::I64 => i64::try_from(x).is_ok(),
        _ => u64::try_from(x).is_ok(),
    };
    let big_endian = v.storage.big_endian;
    let members = e
        .members
        .iter()
        .map(|(n, x)| {
            if !fits(*x) {
                return Err(bad(format!(
                    "{name:?}: enumeration member {n:?} = {x} does not fit {dtype}"
                )));
            }
            let mut b = x
                .to_le_bytes()
                .get(..usize::from(size))
                .unwrap_or_default()
                .to_vec();
            if big_endian {
                b.reverse();
            }
            Ok((n.clone(), b))
        })
        .collect::<Result<Vec<_>>>()?;
    if members.is_empty() {
        return Err(bad(format!("{name:?}: an enumeration needs members")));
    }
    Ok(Datatype::Enum {
        base: Box::new(Datatype::Int {
            size,
            signed,
            big_endian,
        }),
        members,
    })
}

/// The compound types in `t` (itself included), each nested type before any that contains
/// it, without repeats.
fn compound_types(t: &Datatype, out: &mut Vec<Datatype>) {
    match t {
        Datatype::Compound { members, .. } => {
            for m in members {
                compound_types(&m.dtype, out);
            }
            if !out.contains(t) {
                out.push(t.clone());
            }
        }
        Datatype::ArrayOf { base, .. } => compound_types(base, out),
        _ => {}
    }
}

fn merge_ragged(out: &mut RawDataset<'_>) -> Result<HashMap<Name, ArrayD<u64>>> {
    let pairs: Vec<(Name, Name)> = out
        .vars
        .iter()
        .filter_map(|(n, v)| {
            let c = v.storage.ragged.as_ref()?;
            (v.dims.len() == 1 && out.vars.contains_key(c)).then(|| (n.clone(), c.clone()))
        })
        .collect();
    let mut counts = HashMap::new();
    for (values, count) in pairs {
        let (Some(c), Some(v)) = (out.vars.shift_remove(&count), out.vars.get_mut(&values)) else {
            continue;
        };
        let lens: Vec<u64> = crate::cf::num::to_f64(&*c.data.values()?)
            .ok_or_else(|| bad(format!("{count:?}: counts must be integers")))?
            .iter()
            .map(|x| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                (x.fract() == 0.0 && *x >= 0.0)
                    .then_some(*x as u64)
                    .ok_or_else(|| bad(format!("{count:?}: counts must be whole and non-negative")))
            })
            .collect::<Result<_>>()?;
        let total: u64 = lens.iter().copied().fold(0, u64::saturating_add);
        if usize::try_from(total).ok() != v.data.shape().first().copied() {
            return Err(bad(format!(
                "{values:?}: its counts ({count:?}) add up to {total}, not to its length"
            )));
        }
        let obs = v.dims.first().cloned();
        v.dims.clone_from(&c.dims);
        let shaped = ArrayD::from_shape_vec(IxDyn(c.data.shape()), lens)
            .map_err(|e| bad(format!("{count:?}: {e}")))?;
        counts.insert(values.clone(), shaped);
        // The sample dimension goes unless another variable uses it.
        if let Some(obs) = obs
            && !out.vars.values().any(|x| x.dims.contains(&obs))
        {
            out.dims.shift_remove(&obs);
        }
    }
    Ok(counts)
}

fn build_group<'a>(
    ds: &RawDataset<'a>,
    path: &str,
    parent: &WScope,
    next_id: &mut i32,
) -> Result<WGroup<'a>> {
    let merged = merge(ds)?;
    let nothing = Merged::default();
    let (ds, merged) = match &merged {
        Some((d, m)) => (d, m),
        None => (ds, &nothing),
    };
    let ragged_counts = &merged.counts;
    let mut scope = parent.clone();
    // Each dimension's scale is its coordinate variable if there is one, else a dimension-only
    // dataset of the same name.
    let is_coord = |dim: &Name| {
        ds.vars
            .get(dim)
            .is_some_and(|v| v.dims.len() == 1 && v.dims.first() == Some(dim))
    };
    for (dim, d) in &ds.dims {
        scope
            .dims
            .insert(dim.clone(), (path_join(path, dim), *next_id, *d));
        *next_id = next_id.saturating_add(1);
    }
    // Dataset names: variables clashing with a dimension (without being its coordinate
    // variable) get netCDF-C's prefix.
    let dataset_name = |name: &Name| -> String {
        let v = ds.vars.get(name);
        let clash = ds.dims.contains_key(name)
            && !v.is_some_and(|v| v.dims.len() == 1 && v.dims.first() == Some(name));
        if clash {
            format!("{NON_COORD}{name}")
        } else {
            name.to_string()
        }
    };

    // Who uses each dimension of this group (for REFERENCE_LIST), as (dataset path, axis).
    let mut users: HashMap<Name, Vec<(String, i32)>> = HashMap::new();
    for (name, v) in &ds.vars {
        for (axis, dim) in v.dims.iter().enumerate() {
            if is_coord(dim) && dim == name {
                continue; // a coordinate variable is the scale itself
            }
            users.entry(dim.clone()).or_default().push((
                path_join(path, &dataset_name(name)),
                i32::try_from(axis).unwrap_or(0),
            ));
        }
    }
    let scale_attrs = |dim: &Name,
                       name_attr: String,
                       id: i32,
                       users: &HashMap<Name, Vec<(String, i32)>>|
     -> Vec<(String, AttrData)> {
        let mut a = vec![
            (
                "CLASS".to_owned(),
                AttrData::CString("DIMENSION_SCALE".into()),
            ),
            ("NAME".to_owned(), AttrData::CString(name_attr)),
            ("_Netcdf4Dimid".to_owned(), AttrData::ScalarI32(id)),
        ];
        if let Some(u) = users.get(dim).filter(|u| !u.is_empty()) {
            a.push((
                "REFERENCE_LIST".to_owned(),
                AttrData::ReferenceIndexPairs(u.clone()),
            ));
        }
        a
    };

    let mut members: Vec<(String, WNode)> = Vec::new();
    // User-defined types first, inner before outer: netCDF-C reads a variable of a compound
    // or enumeration type only when the group defines that type (and each nested one) as a
    // named datatype.
    let taken =
        |n: &str| ds.vars.contains_key(n) || ds.dims.contains_key(n) || ds.groups.contains_key(n);
    let mut enums: HashMap<&Name, Datatype> = HashMap::new();
    let mut types: Vec<(Option<String>, Datatype)> = Vec::new();
    for (name, v) in &ds.vars {
        if let Some(c) = merged.compounds.get(name) {
            let mut nested = Vec::new();
            compound_types(&c.dtype, &mut nested);
            for t in nested {
                if !types.iter().any(|(_, x)| *x == t) {
                    types.push((None, t));
                }
            }
        }
        if let Some(e) = &v.storage.enumeration {
            let t = enum_datatype(name, v, e)?;
            match types.iter().find(|(_, x)| *x == t) {
                Some((n, _)) if n.is_some() && e.name.is_some() && *n != e.name => {
                    return Err(bad(format!(
                        "{name:?}: enumeration {:?} is also named {:?}",
                        e.name.as_deref().unwrap_or_default(),
                        n.as_deref().unwrap_or_default()
                    )));
                }
                Some(_) => {}
                None => types.push((e.name.clone(), t.clone())),
            }
            enums.insert(name, t);
        }
    }
    let mut k = 0usize;
    for (type_name, t) in types {
        let type_name = match type_name {
            Some(n) if taken(&n) => {
                return Err(bad(format!(
                    "type {n:?}: a variable, dimension or group has the same name"
                )));
            }
            Some(n) => n,
            None => {
                let base = if matches!(t, Datatype::Enum { .. }) {
                    "enum"
                } else {
                    "compound"
                };
                let mut n = format!("{base}{k}_t");
                k = k.saturating_add(1);
                while taken(&n) {
                    n.push('_');
                }
                n
            }
        };
        members.push((type_name, WNode::Datatype(t)));
    }
    // Dimension-only scales first, as netCDF-C defines dimensions before variables.
    for (dim, d) in &ds.dims {
        if is_coord(dim) {
            continue;
        }
        let id = scope.dims.get(dim).map_or(0, |x| x.1);
        let mut w = WDataset::new(Array::F32(ArrayD::zeros(IxDyn(&[d.len]))));
        w.skip_data = true;
        w.unlimited = vec![d.unlimited];
        w.attrs = scale_attrs(dim, format!("{DIM_ONLY}{:>10}", d.len), id, &users);
        members.push((dim.to_string(), WNode::Dataset(w)));
    }
    for (name, v) in &ds.vars {
        // Lazy values are read slice by slice as they are written.
        let mut w = match &v.data {
            // A compound variable, packed from its fields.
            _ if merged.compounds.contains_key(name) => {
                let mut w = WDataset::new(Array::I8(ArrayD::zeros(IxDyn(&[0]))));
                w.compound = merged.compounds.get(name).cloned();
                w
            }
            // A ragged array: a variable-length variable, its values in the global heap.
            _ if ragged_counts.contains_key(name) => {
                let mut w = WDataset::new(hdf5_array(v.data.values()?.into_owned())?);
                w.ragged = ragged_counts.get(name).cloned();
                w
            }
            RawData::Lazy(src) if src.dtype() != DType::Str => {
                let dtype = value_type(src.dtype()).ok_or_else(|| {
                    IoError::Unsupported(format!(
                        "element type {} must be CF-encoded before writing",
                        src.dtype()
                    ))
                })?;
                WDataset::from_source(Arc::new(Streamed {
                    source: Arc::clone(src),
                    dtype,
                }))?
            }
            // Borrowed values are written slice by slice straight from the dataset.
            RawData::Borrowed(a) if a.dtype() != DType::Str && a.ndim() > 0 && !a.is_empty() => {
                let dtype = value_type(a.dtype()).ok_or_else(|| {
                    IoError::Unsupported(format!(
                        "element type {} must be CF-encoded before writing",
                        a.dtype()
                    ))
                })?;
                WDataset::from_source(Arc::new(Viewed {
                    data: a.clone(),
                    dtype,
                }))?
            }
            _ => WDataset::new(hdf5_array(v.data.values()?.into_owned())?),
        };
        w.char_data = v.is_char;
        w.enumeration = enums.get(name).cloned();
        w.unlimited = v
            .dims
            .iter()
            .map(|d| scope.dims.get(d).is_some_and(|x| x.2.unlimited))
            .collect();
        w.chunks.clone_from(&v.storage.chunks);
        w.deflate = v.storage.deflate;
        w.compression = v.storage.compression;
        w.shuffle = v.storage.shuffle;
        w.fletcher32 = v.storage.fletcher32;
        w.half = v.storage.float16;
        // Characters and strings have no byte order.
        w.big_endian = v.storage.big_endian && !v.is_char && v.data.dtype() != DType::Str;
        let needs_chunks = w.unlimited.iter().any(|u| *u)
            || w.deflate.is_some()
            || w.shuffle
            || w.compression.is_some()
            || w.fletcher32;
        // `contiguous` (often just how a file being rewritten was stored) gives way to chunk
        // shapes, filters and unlimited dimensions; asking for both in one explicit encoding
        // is refused where the encoding is given.
        if w.chunks.is_none() && needs_chunks && !v.dims.is_empty() {
            w.chunks = Some(default_chunks(
                match (&w.compound, &w.ragged) {
                    (Some(c), _) => c.shape.as_slice(),
                    (_, Some(r)) => r.shape(),
                    _ => v.data.shape(),
                },
                &w.unlimited,
                element_size(v.data.dtype()),
            ));
        }
        // A variable-length element is just empty: netCDF gives such variables no fill value.
        w.fill = if w.ragged.is_some() || w.compound.is_some() {
            None
        } else {
            v.attrs
                .get("_FillValue")
                .and_then(|f| fill_bytes(f, v.data.dtype()))
        };
        let mut attrs: Vec<(String, AttrData)> = Vec::new();
        let ids = v
            .dims
            .iter()
            .map(|d| {
                scope
                    .dims
                    .get(d)
                    .map(|x| x.1)
                    .ok_or_else(|| bad(format!("{name:?}: unknown dimension {d:?}")))
            })
            .collect::<Result<Vec<i32>>>()?;
        if !ids.is_empty() {
            attrs.push((
                "_Netcdf4Coordinates".into(),
                AttrData::Array(Array::I32(ndarray::Array1::from(ids).into_dyn())),
            ));
        }
        attrs.extend(
            v.attrs
                .iter()
                .filter(|(k, _)| !HIDDEN.contains(&k.as_str()))
                .filter(|(k, _)| {
                    (w.ragged.is_none() && w.compound.is_none()) || k.as_str() != "_FillValue"
                })
                .map(|(k, a)| {
                    // A half-precision variable's fill and missing values are halves too.
                    let half = w.half && matches!(k.as_str(), "_FillValue" | "missing_value");
                    let data = match a {
                        #[allow(clippy::cast_possible_truncation)]
                        AttrValue::F64(x) if half => {
                            AttrData::Half(x.iter().map(|v| *v as f32).collect())
                        }
                        AttrValue::F32(x) if half => AttrData::Half(x.clone()),
                        _ => attr_data(a),
                    };
                    (k.to_string(), data)
                }),
        );
        if v.dims.len() == 1 && is_coord(name) && v.dims.first() == Some(name) {
            let id = scope.dims.get(name).map_or(0, |x| x.1);
            attrs.extend(scale_attrs(name, name.to_string(), id, &users));
        } else if !v.dims.is_empty() {
            let lists = v
                .dims
                .iter()
                .map(|d| {
                    scope
                        .dims
                        .get(d)
                        .map(|x| vec![x.0.clone()])
                        .unwrap_or_default()
                })
                .collect();
            attrs.push(("DIMENSION_LIST".into(), AttrData::ReferenceLists(lists)));
        }
        w.attrs = attrs;
        members.push((dataset_name(name), WNode::Dataset(w)));
    }
    for (name, child) in &ds.groups {
        let p = path_join(path, name);
        members.push((
            name.to_string(),
            WNode::Group(build_group(child, &p, &scope, next_id)?),
        ));
    }
    let attrs = ds
        .attrs
        .iter()
        .filter(|(k, _)| !HIDDEN.contains(&k.as_str()))
        .map(|(k, a)| (k.to_string(), attr_data(a)))
        .collect();
    Ok(WGroup { attrs, members })
}

/// Serialise a raw dataset as a netCDF-4 file in memory.
///
/// # Errors
/// [`IoError::Format`] for a non-netCDF-4 format or a classic-model violation; HDF5 writer
/// errors otherwise.
pub fn write_bytes(ds: &RawDataset, opts: &WriteOptions) -> Result<Vec<u8>> {
    Ok(annoray_hdf5::write_file(&build_root(ds, opts)?)?)
}

/// The HDF5 tree of a netCDF-4 dataset.
fn build_root<'a>(ds: &RawDataset<'a>, opts: &WriteOptions) -> Result<WGroup<'a>> {
    let format = opts.format;
    match format {
        Format::Netcdf4 => {}
        Format::Netcdf4Classic => {
            if !ds.groups.is_empty() {
                return Err(IoError::Format {
                    what: "groups".into(),
                    format,
                    reason: "the classic model has no groups".into(),
                });
            }
            let strs = |a: &Attrs| {
                a.iter()
                    .find(|(_, v)| matches!(v, AttrValue::Strs(_)))
                    .map(|(k, _)| k.clone())
            };
            if let Some(k) =
                strs(&ds.attrs).or_else(|| ds.vars.values().find_map(|v| strs(&v.attrs)))
            {
                return Err(IoError::Format {
                    what: format!("attribute {k:?}"),
                    format,
                    reason: "string lists need the full netCDF-4 model".into(),
                });
            }
            if let Some((n, v)) = ds
                .vars
                .iter()
                .find(|(_, v)| !format.supports(v.data.dtype()) && !v.is_char)
            {
                return Err(IoError::Format {
                    what: format!("variable {n:?} ({})", v.data.dtype()),
                    format,
                    reason: "type not in the classic model".into(),
                });
            }
        }
        other => {
            return Err(IoError::Format {
                what: "file".into(),
                format: other,
                reason: "not a netCDF-4 format".into(),
            });
        }
    }
    let mut next_id = 0;
    let mut root = build_group(ds, "/", &WScope::default(), &mut next_id)?;
    let props = format!("version=2,annoray={}", env!("CARGO_PKG_VERSION"));
    root.attrs
        .insert(0, ("_NCProperties".into(), AttrData::Text(props)));
    if format == Format::Netcdf4Classic {
        root.attrs.insert(
            1,
            (
                "_nc3_strict".into(),
                AttrData::Array(Array::I32(ndarray::arr1(&[1]).into_dyn())),
            ),
        );
    }
    Ok(root)
}

/// Write `ds` as a netCDF-4 file, leaving the values of the root group's `deferred`
/// variables unwritten (fill values until written in place: see
/// [`NetcdfWriter`](super::NetcdfWriter)).
pub(crate) fn write_deferred(
    ds: &RawDataset<'_>,
    path: &Path,
    opts: &WriteOptions,
    deferred: &[Name],
) -> Result<()> {
    let mut root = build_root(ds, opts)?;
    for (name, node) in &mut root.members {
        let var = name.strip_prefix(NON_COORD).unwrap_or(name);
        if let WNode::Dataset(w) = node
            && deferred.iter().any(|d| d.as_str() == var)
        {
            // The source (if any) still gives the shape; it is not read.
            w.skip_data = true;
        }
    }
    super::replace::write_replacing(path, NAME, |out| {
        annoray_hdf5::write_to(&root, out)?;
        Ok(())
    })
}

/// `ds` with its loaded, non-empty numeric arrays behind in-memory sources (recursively).
fn held(mut ds: RawDataset<'_>) -> RawDataset<'_> {
    ds.vars = ds
        .vars
        .into_iter()
        .map(|(k, mut v)| {
            v.data = match v.data {
                RawData::Loaded(a)
                    if a.ndim() > 0
                        && !a.is_empty()
                        && a.dtype() != DType::Str
                        && value_type(a.dtype()).is_some() =>
                {
                    RawData::Lazy(Arc::new(super::raw::Held(a)))
                }
                other => other,
            };
            (k, v)
        })
        .collect();
    ds.groups = ds.groups.into_iter().map(|(k, g)| (k, held(g))).collect();
    ds
}

impl Backend for Netcdf4 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset<'static>> {
        read_group(&File::open(path)?.root()?, &Scope::default(), false)
    }

    fn read_lazy(&self, path: &Path) -> Result<RawDataset<'static>> {
        read_group(&File::open(path)?.root()?, &Scope::default(), true)
    }

    /// Loaded numeric variables are written slice by slice from where they are, not copied.
    fn write_owned(&self, ds: RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        self.write(&held(ds), path, opts)
    }

    /// Writes straight to the file, reading lazy variables slice by slice.
    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let root = build_root(ds, opts)?;
        // A new file beside the old one, renamed over it: datasets still reading the old one
        // (this one, written back to its own file, included) keep it.
        super::replace::write_replacing(path, NAME, |out| {
            annoray_hdf5::write_to(&root, out)?;
            Ok(())
        })
    }
}
