//! netCDF backend for annoray over the C libraries (netCDF-C and HDF5, built from source by
//! the `netcdf` crate's `static` feature).
//!
//! Reads and writes netCDF-3 (classic, 64-bit offset, 64-bit data) and netCDF-4 files,
//! including groups. It is the reference implementation that the pure-Rust backends are tested
//! against.

#![deny(missing_docs)]

use std::path::Path;

use annoray_core::{AttrValue, Attrs, DType, Name, TypedArray};
use annoray_io::{
    Backend, Format, IoError, RawDataset, RawDim, RawVariable, Result, Storage, WriteOptions,
};
use ndarray::{ArrayD, Dimension, IxDyn};
use netcdf::types::{FloatType, IntType, NcTypeDescriptor, NcVariableType};
use netcdf::{AttributeValue, Extent, Extents};

/// The netCDF-C backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct NetcdfC;

const NAME: &str = "netcdf-c";

fn nc_err(e: netcdf::Error) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: e.to_string(),
    }
}

fn vanished(name: &str) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: format!("variable {name:?} not found after definition"),
    }
}

fn shape_err(e: ndarray::ShapeError) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: format!("unexpected data size: {e}"),
    }
}

/// A text character, for writing `NC_CHAR` data.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NcChar(u8);

// SAFETY: `NcChar` is `repr(transparent)` over one byte, exactly the in-memory layout of
// netCDF's `NC_CHAR`, which is all `NcTypeDescriptor` requires.
#[allow(unsafe_code)]
unsafe impl NcTypeDescriptor for NcChar {
    fn type_descriptor() -> NcVariableType {
        NcVariableType::Char
    }
}

// ----- attributes -----

fn attr_from_nc(v: AttributeValue) -> AttrValue {
    match v {
        AttributeValue::Uchar(x) => AttrValue::U8(vec![x]),
        AttributeValue::Uchars(x) => AttrValue::U8(x),
        AttributeValue::Schar(x) => AttrValue::I8(vec![x]),
        AttributeValue::Schars(x) => AttrValue::I8(x),
        AttributeValue::Ushort(x) => AttrValue::U16(vec![x]),
        AttributeValue::Ushorts(x) => AttrValue::U16(x),
        AttributeValue::Short(x) => AttrValue::I16(vec![x]),
        AttributeValue::Shorts(x) => AttrValue::I16(x),
        AttributeValue::Uint(x) => AttrValue::U32(vec![x]),
        AttributeValue::Uints(x) => AttrValue::U32(x),
        AttributeValue::Int(x) => AttrValue::I32(vec![x]),
        AttributeValue::Ints(x) => AttrValue::I32(x),
        AttributeValue::Ulonglong(x) => AttrValue::U64(vec![x]),
        AttributeValue::Ulonglongs(x) => AttrValue::U64(x),
        AttributeValue::Longlong(x) => AttrValue::I64(vec![x]),
        AttributeValue::Longlongs(x) => AttrValue::I64(x),
        AttributeValue::Float(x) => AttrValue::F32(vec![x]),
        AttributeValue::Floats(x) => AttrValue::F32(x),
        AttributeValue::Double(x) => AttrValue::F64(vec![x]),
        AttributeValue::Doubles(x) => AttrValue::F64(x),
        AttributeValue::Str(x) => AttrValue::Str(x),
        AttributeValue::Strs(x) => AttrValue::Strs(x),
    }
}

fn attr_to_nc(v: &AttrValue) -> Result<AttributeValue> {
    Ok(match v {
        AttrValue::Str(x) => AttributeValue::Str(x.clone()),
        AttrValue::Strs(x) => AttributeValue::Strs(x.clone()),
        AttrValue::I8(x) => AttributeValue::Schars(x.clone()),
        AttrValue::I16(x) => AttributeValue::Shorts(x.clone()),
        AttrValue::I32(x) => AttributeValue::Ints(x.clone()),
        AttrValue::I64(x) => AttributeValue::Longlongs(x.clone()),
        AttrValue::U8(x) => AttributeValue::Uchars(x.clone()),
        AttrValue::U16(x) => AttributeValue::Ushorts(x.clone()),
        AttrValue::U32(x) => AttributeValue::Uints(x.clone()),
        AttrValue::U64(x) => AttributeValue::Ulonglongs(x.clone()),
        AttrValue::F32(x) => AttributeValue::Floats(x.clone()),
        AttrValue::F64(x) => AttributeValue::Doubles(x.clone()),
        other => return Err(IoError::Unsupported(format!("attribute value {other:?}"))),
    })
}

fn read_attrs<'a>(attrs: impl Iterator<Item = netcdf::Attribute<'a>>) -> Result<Attrs> {
    attrs
        .map(|a| {
            Ok((
                Name::from(a.name()),
                attr_from_nc(a.value().map_err(nc_err)?),
            ))
        })
        .collect()
}

// ----- reading -----

fn read_var(v: &netcdf::Variable<'_>) -> Result<std::result::Result<RawVariable, String>> {
    let dims: Vec<Name> = v
        .dimensions()
        .iter()
        .map(|d| Name::from(d.name()))
        .collect();
    let shape: Vec<usize> = v.dimensions().iter().map(netcdf::Dimension::len).collect();
    let mut is_char = false;
    macro_rules! get {
        ($var:ident, $t:ty) => {
            TypedArray::$var(v.get::<$t, _>(..).map_err(nc_err)?)
        };
    }
    let data = match v.vartype() {
        NcVariableType::Int(IntType::I8) => get!(I8, i8),
        NcVariableType::Int(IntType::I16) => get!(I16, i16),
        NcVariableType::Int(IntType::I32) => get!(I32, i32),
        NcVariableType::Int(IntType::I64) => get!(I64, i64),
        NcVariableType::Int(IntType::U8) => get!(U8, u8),
        NcVariableType::Int(IntType::U16) => get!(U16, u16),
        NcVariableType::Int(IntType::U32) => get!(U32, u32),
        NcVariableType::Int(IntType::U64) => get!(U64, u64),
        NcVariableType::Float(FloatType::F32) => get!(F32, f32),
        NcVariableType::Float(FloatType::F64) => get!(F64, f64),
        NcVariableType::Char => {
            is_char = true;
            let bytes = v.get_raw_values(..).map_err(nc_err)?;
            TypedArray::U8(ArrayD::from_shape_vec(IxDyn(&shape), bytes).map_err(shape_err)?)
        }
        NcVariableType::String => {
            let mut out = Vec::with_capacity(v.len());
            for idx in ndarray::indices(IxDyn(&shape)) {
                let ext: Vec<Extent> = idx.slice().iter().map(|i| Extent::from(*i)).collect();
                let ext = if ext.is_empty() {
                    Extents::All
                } else {
                    Extents::from(ext)
                };
                out.push(v.get_string(ext).map_err(nc_err)?);
            }
            TypedArray::Str(ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(shape_err)?)
        }
        other => return Ok(Err(format!("unsupported netCDF type {other:?}"))),
    };
    let mut storage = Storage::default();
    storage.chunks = v.chunking().map_err(nc_err)?;
    Ok(Ok(RawVariable {
        dims,
        data: data.into(),
        is_char,
        attrs: read_attrs(v.attributes())?,
        storage,
    }))
}

fn assemble<'a>(
    dims: impl Iterator<Item = netcdf::Dimension<'a>>,
    vars: impl Iterator<Item = netcdf::Variable<'a>>,
    attrs: impl Iterator<Item = netcdf::Attribute<'a>>,
) -> Result<RawDataset> {
    let mut ds = RawDataset {
        dims: dims
            .map(|d| {
                (
                    Name::from(d.name()),
                    RawDim {
                        len: d.len(),
                        unlimited: d.is_unlimited(),
                    },
                )
            })
            .collect(),
        attrs: read_attrs(attrs)?,
        ..RawDataset::default()
    };
    for v in vars {
        let name = Name::from(v.name());
        match read_var(&v)? {
            Ok(raw) => {
                ds.vars.insert(name, raw);
            }
            Err(reason) => ds.skipped.push((name, reason)),
        }
    }
    Ok(ds)
}

fn read_group(g: &netcdf::Group<'_>) -> Result<RawDataset> {
    let mut ds = assemble(g.dimensions(), g.variables(), g.attributes())?;
    for child in g.groups() {
        ds.groups
            .insert(Name::from(child.name()), read_group(&child)?);
    }
    Ok(ds)
}

// ----- writing -----

/// `add_variable::<T>` for the Rust type matching `d` (the generic form has the right
/// lifetimes on both files and groups).
macro_rules! add_typed {
    ($t:expr, $name:expr, $dims:expr, $d:expr, $is_char:expr) => {{
        let r = match ($d, $is_char) {
            (DType::U8, true) => $t.add_variable::<NcChar>($name, $dims).map(drop),
            (DType::I8, _) => $t.add_variable::<i8>($name, $dims).map(drop),
            (DType::I16, _) => $t.add_variable::<i16>($name, $dims).map(drop),
            (DType::I32, _) => $t.add_variable::<i32>($name, $dims).map(drop),
            (DType::I64, _) => $t.add_variable::<i64>($name, $dims).map(drop),
            (DType::U8, false) => $t.add_variable::<u8>($name, $dims).map(drop),
            (DType::U16, _) => $t.add_variable::<u16>($name, $dims).map(drop),
            (DType::U32, _) => $t.add_variable::<u32>($name, $dims).map(drop),
            (DType::U64, _) => $t.add_variable::<u64>($name, $dims).map(drop),
            (DType::F32, _) => $t.add_variable::<f32>($name, $dims).map(drop),
            (DType::F64, _) => $t.add_variable::<f64>($name, $dims).map(drop),
            (other, _) => {
                return Err(IoError::Unsupported(format!(
                    "element type {other} must be CF-encoded before writing"
                )));
            }
        };
        r.map_err(nc_err)
    }};
}

/// Something that can hold dimensions, variables and attributes: a classic file or a group.
trait Target {
    fn dim(&mut self, name: &str, d: RawDim) -> netcdf::Result<()>;
    fn var(&mut self, name: &str, dims: &[&str], d: DType, is_char: bool) -> Result<()>;
    fn string_var(&mut self, name: &str, dims: &[&str]) -> netcdf::Result<()>;
    fn attr(&mut self, name: &str, v: AttributeValue) -> netcdf::Result<()>;
    fn var_mut(&mut self, name: &str) -> Option<netcdf::VariableMut<'_>>;
}

impl Target for netcdf::FileMut {
    fn dim(&mut self, name: &str, d: RawDim) -> netcdf::Result<()> {
        if d.unlimited {
            self.add_unlimited_dimension(name).map(drop)
        } else {
            self.add_dimension(name, d.len).map(drop)
        }
    }
    fn var(&mut self, name: &str, dims: &[&str], d: DType, is_char: bool) -> Result<()> {
        add_typed!(self, name, dims, d, is_char)
    }
    fn string_var(&mut self, name: &str, dims: &[&str]) -> netcdf::Result<()> {
        self.add_string_variable(name, dims).map(drop)
    }
    fn attr(&mut self, name: &str, v: AttributeValue) -> netcdf::Result<()> {
        self.add_attribute(name, v).map(drop)
    }
    fn var_mut(&mut self, name: &str) -> Option<netcdf::VariableMut<'_>> {
        self.variable_mut(name)
    }
}

impl Target for netcdf::GroupMut<'_> {
    fn dim(&mut self, name: &str, d: RawDim) -> netcdf::Result<()> {
        if d.unlimited {
            self.add_unlimited_dimension(name).map(drop)
        } else {
            self.add_dimension(name, d.len).map(drop)
        }
    }
    fn var(&mut self, name: &str, dims: &[&str], d: DType, is_char: bool) -> Result<()> {
        add_typed!(self, name, dims, d, is_char)
    }
    fn string_var(&mut self, name: &str, dims: &[&str]) -> netcdf::Result<()> {
        self.add_string_variable(name, dims).map(drop)
    }
    fn attr(&mut self, name: &str, v: AttributeValue) -> netcdf::Result<()> {
        self.add_attribute(name, v).map(drop)
    }
    fn var_mut(&mut self, name: &str) -> Option<netcdf::VariableMut<'_>> {
        self.variable_mut(name)
    }
}

fn check_format(name: &Name, v: &RawVariable, format: Format) -> Result<()> {
    let d = v.data.dtype();
    let ok = (v.is_char && d == DType::U8) || format.supports(d);
    if ok {
        Ok(())
    } else {
        Err(IoError::Format {
            what: format!("variable {name:?} ({d})"),
            format,
            reason: "type not available".into(),
        })
    }
}

fn define(t: &mut impl Target, ds: &RawDataset, format: Format) -> Result<()> {
    for (name, d) in &ds.dims {
        t.dim(name, *d).map_err(nc_err)?;
    }
    for (k, v) in &ds.attrs {
        t.attr(k, attr_to_nc(v)?).map_err(nc_err)?;
    }
    for (name, v) in &ds.vars {
        check_format(name, v, format)?;
        let dims: Vec<&str> = v.dims.iter().map(Name::as_str).collect();
        if v.data.dtype() == DType::Str {
            t.string_var(name, &dims).map_err(nc_err)?;
        } else {
            t.var(name, &dims, v.data.dtype(), v.is_char)?;
        }
        let mut var = t.var_mut(name).ok_or_else(|| vanished(name))?;
        if !format.is_netcdf3() && !dims.is_empty() {
            if let Some(chunks) = &v.storage.chunks {
                var.set_chunking(chunks).map_err(nc_err)?;
            }
            if let Some(level) = v.storage.deflate {
                var.set_compression(i32::from(level), v.storage.shuffle)
                    .map_err(nc_err)?;
            }
        }
        // `_FillValue` must be defined before any data is written.
        if let Some(fill) = v.attrs.get("_FillValue") {
            var.put_attribute("_FillValue", attr_to_nc(fill)?)
                .map_err(nc_err)?;
        }
        for (k, a) in v.attrs.iter().filter(|(k, _)| *k != "_FillValue") {
            var.put_attribute(k, attr_to_nc(a)?).map_err(nc_err)?;
        }
    }
    Ok(())
}

fn put_data(var: &mut netcdf::VariableMut<'_>, v: &RawVariable) -> Result<()> {
    let shape = v.data.shape().to_vec();
    let ext: Extents = if shape.is_empty() {
        Extents::All
    } else {
        Extents::from(
            shape
                .iter()
                .map(|n| Extent::from(0..*n))
                .collect::<Vec<_>>(),
        )
    };
    if shape.contains(&0) {
        return Ok(());
    }
    macro_rules! put {
        ($a:expr) => {
            var.put_values(&$a.iter().copied().collect::<Vec<_>>(), ext)
                .map_err(nc_err)?
        };
    }
    let values = v.data.values()?;
    match &*values {
        TypedArray::U8(a) if v.is_char => {
            var.put_values(&a.iter().map(|b| NcChar(*b)).collect::<Vec<_>>(), ext)
                .map_err(nc_err)?;
        }
        TypedArray::I8(a) => put!(a),
        TypedArray::I16(a) => put!(a),
        TypedArray::I32(a) => put!(a),
        TypedArray::I64(a) => put!(a),
        TypedArray::U8(a) => put!(a),
        TypedArray::U16(a) => put!(a),
        TypedArray::U32(a) => put!(a),
        TypedArray::U64(a) => put!(a),
        TypedArray::F32(a) => put!(a),
        TypedArray::F64(a) => put!(a),
        TypedArray::Str(a) => {
            for (idx, s) in a.indexed_iter() {
                let e: Vec<Extent> = idx.slice().iter().map(|i| Extent::from(*i)).collect();
                let e = if e.is_empty() {
                    Extents::All
                } else {
                    Extents::from(e)
                };
                var.put_string(s, e).map_err(nc_err)?;
            }
        }
        other => {
            return Err(IoError::Unsupported(format!(
                "element type {} must be CF-encoded first",
                other.dtype()
            )));
        }
    }
    Ok(())
}

fn write_data(t: &mut impl Target, ds: &RawDataset) -> Result<()> {
    for (name, v) in &ds.vars {
        let mut var = t.var_mut(name).ok_or_else(|| vanished(name))?;
        put_data(&mut var, v)?;
    }
    Ok(())
}

fn define_groups(g: &mut netcdf::GroupMut<'_>, ds: &RawDataset, format: Format) -> Result<()> {
    define(g, ds, format)?;
    for (name, child) in &ds.groups {
        let mut cg = g.add_group(name).map_err(nc_err)?;
        define_groups(&mut cg, child, format)?;
    }
    Ok(())
}

fn write_groups(g: &mut netcdf::GroupMut<'_>, ds: &RawDataset) -> Result<()> {
    write_data(g, ds)?;
    for (name, child) in &ds.groups {
        let mut cg = g.group_mut(name).ok_or_else(|| IoError::Backend {
            backend: NAME,
            message: format!("group {name:?} vanished"),
        })?;
        write_groups(&mut cg, child)?;
    }
    Ok(())
}

fn options(format: Format) -> netcdf::Options {
    use netcdf::Options as O;
    match format {
        Format::Netcdf4 => O::NETCDF4,
        Format::Netcdf4Classic => O::NETCDF4 | O::CLASSIC,
        Format::Netcdf3Offset64 => O::_64BIT_OFFSET,
        Format::Netcdf3Data64 => O::_64BIT_DATA,
        _ => O::empty(),
    }
}

impl Backend for NetcdfC {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        let file = netcdf::open(path).map_err(nc_err)?;
        match file.root() {
            Some(root) => read_group(&root),
            None => assemble(file.dimensions(), file.variables(), file.attributes()),
        }
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let format = opts.format;
        if format.is_classic_model() && !ds.groups.is_empty() {
            return Err(IoError::Format {
                what: "groups".into(),
                format,
                reason: "classic model has no groups".into(),
            });
        }
        let mut file = netcdf::create_with(path, options(format)).map_err(nc_err)?;
        match file.root_mut() {
            Some(mut root) => define_groups(&mut root, ds, format)?,
            None => define(&mut file, ds, format)?,
        }
        file.enddef().map_err(nc_err)?;
        match file.root_mut() {
            Some(mut root) => write_groups(&mut root, ds)?,
            None => write_data(&mut file, ds)?,
        }
        file.close().map_err(nc_err)
    }
}
