//! Pure-Rust netCDF-4 for xarray-rs: the netCDF data model on top of HDF5 (`xr-hdf5`), read and
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

use xr_core::{AttrValue, Attrs, DType, Name, Owned, TypedArray};
use xr_hdf5::{AttrData, Dataset, Datatype, File, Group, Node, WDataset, WGroup, WNode};
use xr_io::{
    Backend, Format, IoError, RawDataset, RawDim, RawVariable, Result, Storage, WriteOptions,
};

/// The pure-Rust netCDF-4 backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Netcdf4;

const NAME: &str = "netcdf4";
const NON_COORD: &str = "_nc4_non_coord_";
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
        .and_then(|a| match xr_hdf5::attr_value(&a) {
            Some(AttrValue::Str(s)) => Some(s),
            _ => None,
        }))
}

fn int_attr(d: &Dataset, name: &str) -> Result<Option<i32>> {
    Ok(d.attrs()?
        .into_iter()
        .find(|a| a.name == name)
        .and_then(|a| match xr_hdf5::attr_value(&a) {
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
    attrs: Vec<xr_hdf5::Attribute>,
    skipped: &mut Vec<(Name, String)>,
    owner: &str,
) -> Attrs {
    let mut out = Attrs::new();
    for a in attrs {
        if HIDDEN.contains(&a.name.as_str()) {
            continue;
        }
        match xr_hdf5::attr_value(&a) {
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

fn read_group(g: &Group, parent: &Scope) -> Result<RawDataset> {
    let mut ds = RawDataset::default();
    let mut scope = parent.clone();
    let members = g.members()?;

    // Dimensions: the scales of this group, ordered by dimension id.
    let mut scales: Vec<(u64, Scale)> = Vec::new();
    for (name, node) in &members {
        if let Node::Dataset(d) = node {
            if let Some(s) = scale_of(name, d)? {
                scales.push((d.address(), s));
            }
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

    for (name, node) in members {
        match node {
            Node::Group(child) => {
                ds.groups
                    .insert(Name::from(name.as_str()), read_group(&child, &scope)?);
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
                    .map(xr_hdf5::Attribute::reference_lists)
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
                    .and_then(xr_hdf5::attr_value)
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
                let is_char = matches!(d.dtype(), Datatype::FixedString { size: 1, .. });
                let data = match d.read() {
                    Ok(v) if is_char => chars(v),
                    Ok(v) => v,
                    Err(e) => {
                        ds.skipped.push((var_name, e.to_string()));
                        continue;
                    }
                };
                let mut storage = Storage::default();
                storage.chunks = d.chunks();
                (storage.deflate, storage.shuffle) = d.compression();
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
pub fn read_bytes(bytes: Vec<u8>) -> Result<RawDataset> {
    read_group(&File::from_bytes(bytes)?.root()?, &Scope::default())
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

fn element_size(d: &TypedArray<Owned>) -> usize {
    match d {
        TypedArray::I8(_) | TypedArray::U8(_) => 1,
        TypedArray::I16(_) | TypedArray::U16(_) => 2,
        TypedArray::I32(_) | TypedArray::U32(_) | TypedArray::F32(_) => 4,
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

fn build_group(ds: &RawDataset, path: &str, parent: &WScope, next_id: &mut i32) -> Result<WGroup> {
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
    // Dimension-only scales first, as netCDF-C defines dimensions before variables.
    for (dim, d) in &ds.dims {
        if is_coord(dim) {
            continue;
        }
        let id = scope.dims.get(dim).map_or(0, |x| x.1);
        let mut w = WDataset::new(TypedArray::F32(ndarray::ArrayD::zeros(ndarray::IxDyn(&[
            d.len,
        ]))));
        w.skip_data = true;
        w.unlimited = vec![d.unlimited];
        w.attrs = scale_attrs(dim, format!("{DIM_ONLY}{:>10}", d.len), id, &users);
        members.push((dim.to_string(), WNode::Dataset(w)));
    }
    for (name, v) in &ds.vars {
        let mut w = WDataset::new(v.data.clone());
        w.char_data = v.is_char;
        w.unlimited = v
            .dims
            .iter()
            .map(|d| scope.dims.get(d).is_some_and(|x| x.2.unlimited))
            .collect();
        w.chunks.clone_from(&v.storage.chunks);
        w.deflate = v.storage.deflate;
        w.shuffle = v.storage.shuffle;
        let needs_chunks = w.unlimited.iter().any(|u| *u) || w.deflate.is_some() || w.shuffle;
        if w.chunks.is_none() && needs_chunks && !v.dims.is_empty() {
            w.chunks = Some(default_chunks(
                v.data.shape(),
                &w.unlimited,
                element_size(&v.data),
            ));
        }
        w.fill = v
            .attrs
            .get("_FillValue")
            .and_then(|f| fill_bytes(f, v.data.dtype()));
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
                AttrData::Value(AttrValue::I32(ids)),
            ));
        }
        attrs.extend(
            v.attrs
                .iter()
                .filter(|(k, _)| !HIDDEN.contains(&k.as_str()))
                .map(|(k, a)| (k.to_string(), AttrData::Value(a.clone()))),
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
        .map(|(k, a)| (k.to_string(), AttrData::Value(a.clone())))
        .collect();
    Ok(WGroup { attrs, members })
}

/// Serialise a raw dataset as a netCDF-4 file in memory.
///
/// # Errors
/// [`IoError::Format`] for a non-netCDF-4 format or a classic-model violation; HDF5 writer
/// errors otherwise.
pub fn write_bytes(ds: &RawDataset, opts: &WriteOptions) -> Result<Vec<u8>> {
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
    let props = format!("version=2,xarray-rs={}", env!("CARGO_PKG_VERSION"));
    root.attrs.insert(
        0,
        (
            "_NCProperties".into(),
            AttrData::Value(AttrValue::Str(props)),
        ),
    );
    if format == Format::Netcdf4Classic {
        root.attrs.insert(
            1,
            (
                "_nc3_strict".into(),
                AttrData::Value(AttrValue::I32(vec![1])),
            ),
        );
    }
    xr_hdf5::write_file(&root)
}

impl Backend for Netcdf4 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        read_group(&File::open(path)?.root()?, &Scope::default())
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let bytes = write_bytes(ds, opts)?;
        std::fs::write(path, bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
    }
}
