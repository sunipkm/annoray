//! Pure-Rust netCDF-3 for xarray-rs: the classic (CDF-1), 64-bit offset (CDF-2) and 64-bit
//! data (CDF-5) formats, read and written.
//!
//! Files are treated as untrusted input: the parser bounds-checks every read, uses checked
//! arithmetic for every size and offset, and never allocates more than the file could hold, so
//! a corrupt file gives an [`IoError`], never a panic.
//!
//! ```no_run
//! use xr_io::{Backend, Format, WriteOptions};
//! use xr_netcdf3::Netcdf3;
//!
//! # fn main() -> Result<(), xr_io::IoError> {
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

use xr_core::{Attrs, Name, Owned, TypedArray};
use xr_io::{Backend, FileBytes, IoError, RawDataset, RawDim, Result, WriteOptions};

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
    /// Memory-map and parse the header (see [`xr_io::map_file`] for the one caveat).
    ///
    /// # Errors
    /// I/O errors, or [`IoError::Backend`] for malformed headers.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_bytes(xr_io::map_file(path.as_ref())?)
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
}

impl Backend for Netcdf3 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        Netcdf3File::open(path)?.to_raw()
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let bytes = write_bytes(ds, opts)?;
        std::fs::write(path, bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
    }
}
