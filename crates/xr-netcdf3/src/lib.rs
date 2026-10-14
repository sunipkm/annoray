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

use xr_io::{Backend, IoError, RawDataset, Result, WriteOptions};

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

impl Backend for Netcdf3 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        let bytes = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        read_bytes(&bytes)
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        let bytes = write_bytes(ds, opts)?;
        std::fs::write(path, bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
    }
}
