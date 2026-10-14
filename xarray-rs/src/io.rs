//! Reading and writing files.
//!
//! netCDF-3, netCDF-4 and HDF5 are read and written in pure Rust; files are memory-mapped.
//! The optional `netcdf-c` feature adds the C libraries as an alternative backend.
//!
//! ```no_run
//! # fn main() -> Result<(), xarray_rs::Error> {
//! use xarray_rs::io::{Format, ToNetcdf, WriteOptions, open_dataset};
//!
//! let ds = open_dataset("air.nc")?;
//! ds.to_netcdf_with("copy.nc", &WriteOptions::format(Format::Netcdf3Offset64))?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;

pub use xr_cf::{DecodeOptions, EncodeOptions};
pub use xr_io::{Backend, Format, RawDataset, WriteOptions};

use crate::{Dataset, DatasetBase, Error, Repr};

/// Options for opening a dataset.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OpenOptions {
    /// A `/`-separated group path (netCDF-4); the root group by default.
    pub group: Option<String>,
    /// What to CF-decode.
    pub decode: DecodeOptions,
}

impl OpenOptions {
    /// Open a group instead of the root.
    #[must_use]
    pub fn group(mut self, path: impl Into<String>) -> Self {
        self.group = Some(path.into());
        self
    }

    /// Choose what to decode.
    #[must_use]
    pub fn decode(mut self, decode: DecodeOptions) -> Self {
        self.decode = decode;
        self
    }
}

/// A dataset opened from a file, with any non-fatal decoding warnings.
#[derive(Debug)]
pub struct Opened {
    /// The dataset.
    pub dataset: Dataset,
    /// Problems that did not stop decoding (unknown units, undecodable times, skipped types).
    pub warnings: Vec<String>,
}

/// Open a file with a specific backend.
///
/// # Errors
/// I/O, format or decoding errors.
pub fn open_dataset_with_backend(
    backend: &dyn Backend,
    path: impl AsRef<Path>,
    opts: &OpenOptions,
) -> Result<Opened, Error> {
    let raw = backend.read(path.as_ref())?;
    let group = raw.group(opts.group.as_deref().unwrap_or(""))?;
    let decoded = xr_cf::decode(group, &opts.decode)?;
    Ok(Opened {
        dataset: decoded.dataset,
        warnings: decoded.warnings,
    })
}

/// Write a dataset with a specific backend.
///
/// # Errors
/// Encoding, format or I/O errors.
pub fn to_netcdf_with_backend<S: Repr>(
    backend: &dyn Backend,
    ds: &DatasetBase<S>,
    path: impl AsRef<Path>,
    opts: &WriteOptions,
) -> Result<(), Error> {
    let raw = xr_cf::encode(ds, &EncodeOptions::format(opts.format))?;
    backend.write(&raw, path.as_ref(), opts)?;
    Ok(())
}

/// Picks a backend per file, all pure Rust: netCDF-3 via [`xr_netcdf3::Netcdf3`], netCDF-4 and
/// HDF5 via [`xr_netcdf4::Netcdf4`]. Files are memory-mapped, so opening is cheap.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoBackend;

impl Backend for AutoBackend {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn read(&self, path: &Path) -> xr_io::Result<RawDataset> {
        let bytes = xr_io::map_file(path)?;
        if xr_netcdf3::is_netcdf3(&bytes) {
            return xr_netcdf3::Netcdf3File::from_file_bytes(bytes)?.to_raw();
        }
        if xr_hdf5::is_hdf5(&bytes) {
            return xr_netcdf4::Netcdf4.read(path);
        }
        Err(xr_io::IoError::Unsupported(format!(
            "{}: not a netCDF or HDF5 file",
            path.display()
        )))
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> xr_io::Result<()> {
        if opts.format.is_netcdf3() {
            xr_netcdf3::Netcdf3.write(ds, path, opts)
        } else {
            xr_netcdf4::Netcdf4.write(ds, path, opts)
        }
    }
}

/// The netCDF-C/HDF5 C-library backend, for comparison with the pure-Rust one.
#[cfg(feature = "netcdf-c")]
#[must_use]
pub fn c_backend() -> &'static dyn Backend {
    &xr_io_c::NetcdfC
}

/// The default backend ([`AutoBackend`]).
#[must_use]
pub fn default_backend() -> &'static dyn Backend {
    &AutoBackend
}

/// Open a netCDF file (netCDF-3, or netCDF-4 with the `netcdf-c` feature) and CF-decode it.
///
/// # Errors
/// I/O or decoding errors.
pub fn open_dataset(path: impl AsRef<Path>) -> Result<Dataset, Error> {
    Ok(open_dataset_with(path, &OpenOptions::default())?.dataset)
}

/// Open a netCDF file with options, keeping decoding warnings.
///
/// # Errors
/// I/O or decoding errors.
pub fn open_dataset_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<Opened, Error> {
    open_dataset_with_backend(default_backend(), path, opts)
}

/// Writing datasets to netCDF.
pub trait ToNetcdf {
    /// Write with the default format: netCDF-4 with the `netcdf-c` feature, otherwise
    /// netCDF-3 with 64-bit offsets.
    ///
    /// # Errors
    /// Encoding or I/O errors.
    fn to_netcdf(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let format = if cfg!(feature = "netcdf-c") {
            Format::Netcdf4
        } else {
            Format::Netcdf3Offset64
        };
        self.to_netcdf_with(path, &WriteOptions::format(format))
    }

    /// Write with options (e.g. `WriteOptions::format(Format::Netcdf3Offset64)`).
    ///
    /// # Errors
    /// Encoding, format or I/O errors.
    fn to_netcdf_with(&self, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<(), Error>;
}

impl<S: Repr> ToNetcdf for DatasetBase<S> {
    fn to_netcdf_with(&self, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<(), Error> {
        to_netcdf_with_backend(default_backend(), self, path, opts)
    }
}
