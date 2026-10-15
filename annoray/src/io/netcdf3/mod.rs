//! Pure-Rust netCDF-3: the classic (CDF-1), 64-bit offset (CDF-2) and 64-bit
//! data (CDF-5) formats, read and written.
//!
//! Files are treated as untrusted input: the parser bounds-checks every read, uses checked
//! arithmetic for every size and offset, and never allocates more than the file could hold, so
//! a corrupt file gives an [`IoError`], never a panic.
//!
//! ```no_run
//! use annoray::io::netcdf3::Netcdf3;
//! use annoray::io::{Backend, Format, WriteOptions};
//!
//! # fn main() -> Result<(), annoray::io::IoError> {
//! let raw = Netcdf3.read("in.nc".as_ref())?;
//! Netcdf3.write(&raw, "out.nc".as_ref(), &WriteOptions::format(Format::Netcdf3Offset64))?;
//! # Ok(())
//! # }
//! ```

mod ck;
mod read;
mod types;
mod write;

use std::path::Path;
use std::sync::Arc;

use crate::io::{
    Backend, FileBytes, IoError, RawData, RawDataset, RawDim, RawVariable, Result, Storage,
    WriteOptions,
};
use crate::{Attrs, DType, LazySource, Name, Owned, TypedArray};

/// The pure-Rust netCDF-3 backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Netcdf3;

const NAME: &str = "netcdf3";

/// The format version, from the magic number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Version {
    Cdf1,
    Cdf2,
    Cdf5,
}

pub(crate) fn bad(message: impl Into<String>) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: message.into(),
    }
}

/// Whether `bytes` start like a netCDF-3 file.
#[must_use]
pub fn is_netcdf3(bytes: &[u8]) -> bool {
    matches!(bytes.get(..4), Some(b"CDF\x01" | b"CDF\x02" | b"CDF\x05"))
}

/// Parse a netCDF-3 file from memory.
///
/// # Errors
/// [`IoError::Backend`] for anything malformed.
pub fn read_bytes(bytes: &[u8]) -> Result<RawDataset> {
    read::parse(bytes)
}

/// Serialise a dataset as netCDF-3 in memory.
///
/// # Errors
/// [`IoError::Format`] if the dataset needs features netCDF-3 lacks.
pub fn write_bytes(ds: &RawDataset, opts: &WriteOptions) -> Result<Vec<u8>> {
    write::serialize(ds, opts.format)
}

/// An open netCDF-3 file: the header is parsed, data is read on demand straight from the
/// (memory-mapped) file, so opening is cheap and slices touch only what they need.
#[derive(Debug)]
pub struct Netcdf3File {
    bytes: FileBytes,
    header: read::Header,
}

impl Netcdf3File {
    /// Memory-map and parse the header (see [`crate::io::map_file`] for the one caveat).
    ///
    /// # Errors
    /// I/O errors, or [`IoError::Backend`] for malformed headers.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_bytes(crate::io::map_file(path.as_ref())?)
    }

    /// Parse the header of mapped or in-memory bytes.
    ///
    /// # Errors
    /// [`IoError::Backend`] for malformed headers.
    pub fn from_file_bytes(bytes: FileBytes) -> Result<Self> {
        let header = read::parse_header(&bytes)?;
        Ok(Self { bytes, header })
    }

    /// Dimensions, in file order.
    #[must_use]
    pub fn dims(&self) -> &[(Name, RawDim)] {
        &self.header.dims
    }

    /// Global attributes.
    #[must_use]
    pub fn attrs(&self) -> &Attrs {
        &self.header.attrs
    }

    /// Variable names, in file order.
    pub fn variables(&self) -> impl Iterator<Item = &Name> {
        self.header.vars.iter().map(|v| &v.name)
    }

    fn var(&self, name: &str) -> Result<&read::VarInfo> {
        self.header
            .vars
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| bad(format!("no variable {name:?}")))
    }

    /// A variable's dimension names, shape and attributes.
    ///
    /// # Errors
    /// If there is no such variable.
    pub fn variable(&self, name: &str) -> Result<(&[Name], &[usize], &Attrs)> {
        let v = self.var(name)?;
        Ok((&v.dims, &v.shape, &v.attrs))
    }

    /// Read a whole variable.
    ///
    /// # Errors
    /// If there is no such variable or its data is corrupt.
    pub fn read(&self, name: &str) -> Result<TypedArray<Owned>> {
        read::read_var(&self.bytes, &self.header, self.var(name)?)
    }

    /// Read the hyperslab `[start, start + count)` of a variable.
    ///
    /// # Errors
    /// If there is no such variable, the slice is out of range, or the data is corrupt.
    pub fn read_slice(
        &self,
        name: &str,
        start: &[usize],
        count: &[usize],
    ) -> Result<TypedArray<Owned>> {
        read::read_slice(&self.bytes, &self.header, self.var(name)?, start, count)
    }

    /// Everything, as a raw dataset.
    ///
    /// # Errors
    /// If any variable's data is corrupt.
    pub fn to_raw(&self) -> Result<RawDataset> {
        read::to_raw(&self.bytes, &self.header)
    }

    /// Everything, as a raw dataset whose variables read their data on demand.
    #[must_use]
    pub fn to_raw_lazy(self: &Arc<Self>) -> RawDataset {
        let h = &self.header;
        let mut ds = RawDataset {
            attrs: h.attrs.clone(),
            ..RawDataset::default()
        };
        ds.dims.extend(h.dims.iter().cloned());
        for v in &h.vars {
            let source = Nc3Source {
                file: Arc::clone(self),
                name: v.name.to_string(),
                dtype: v.ty.dtype(),
                shape: v.shape.clone(),
            };
            ds.vars.insert(
                v.name.clone(),
                RawVariable {
                    dims: v.dims.clone(),
                    data: RawData::Lazy(Arc::new(source)),
                    is_char: v.ty == types::NcType::Char,
                    attrs: v.attrs.clone(),
                    storage: Storage::default(),
                },
            );
        }
        ds
    }
}

/// One variable of an open file, read slice by slice.
#[derive(Debug)]
struct Nc3Source {
    file: Arc<Netcdf3File>,
    name: String,
    dtype: DType,
    shape: Vec<usize>,
}

impl LazySource for Nc3Source {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> crate::Result<TypedArray<Owned>> {
        self.file
            .read_slice(&self.name, start, count)
            .map_err(|e| crate::CoreError::Read(e.to_string()))
    }
}

impl Backend for Netcdf3 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        Netcdf3File::open(path)?.to_raw()
    }

    fn read_lazy(&self, path: &Path) -> Result<RawDataset> {
        Ok(Arc::new(Netcdf3File::open(path)?).to_raw_lazy())
    }

    /// Writes straight to the file, reading lazy variables a slab at a time.
    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let err = |e: std::io::Error| bad(format!("{}: {e}", path.display()));
        let file = std::fs::File::create(path).map_err(err)?;
        let mut out = std::io::BufWriter::new(file);
        write::write_to(ds, opts.format, &mut out)?;
        out.into_inner()
            .map_err(|e| err(e.into_error()))?
            .sync_all()
            .map_err(err)
    }
}
