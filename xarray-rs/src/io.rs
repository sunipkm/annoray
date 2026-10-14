//! Reading and writing files.
//!
//! ```no_run
//! # #[cfg(feature = "netcdf-c")]
//! # fn main() -> Result<(), xarray_rs::Error> {
//! use xarray_rs::io::{ToNetcdf, open_dataset};
//!
//! let ds = open_dataset("air.nc")?;
//! ds.to_netcdf("copy.nc")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "netcdf-c"))]
//! # fn main() {}
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

/// The default backend.
#[cfg(feature = "netcdf-c")]
#[must_use]
pub fn default_backend() -> &'static dyn Backend {
    &xr_io_c::NetcdfC
}

/// Open a netCDF file (netCDF-3 or netCDF-4) and CF-decode it.
///
/// # Errors
/// I/O or decoding errors.
#[cfg(feature = "netcdf-c")]
pub fn open_dataset(path: impl AsRef<Path>) -> Result<Dataset, Error> {
    Ok(open_dataset_with(path, &OpenOptions::default())?.dataset)
}

/// Open a netCDF file with options, keeping decoding warnings.
///
/// # Errors
/// I/O or decoding errors.
#[cfg(feature = "netcdf-c")]
pub fn open_dataset_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<Opened, Error> {
    open_dataset_with_backend(default_backend(), path, opts)
}

/// Writing datasets to netCDF.
#[cfg(feature = "netcdf-c")]
pub trait ToNetcdf {
    /// Write as netCDF-4.
    ///
    /// # Errors
    /// Encoding or I/O errors.
    fn to_netcdf(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.to_netcdf_with(path, &WriteOptions::default())
    }

    /// Write with options (e.g. `WriteOptions::format(Format::Netcdf3Offset64)`).
    ///
    /// # Errors
    /// Encoding, format or I/O errors.
    fn to_netcdf_with(&self, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<(), Error>;
}

#[cfg(feature = "netcdf-c")]
impl<S: Repr> ToNetcdf for DatasetBase<S> {
    fn to_netcdf_with(&self, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<(), Error> {
        to_netcdf_with_backend(default_backend(), self, path, opts)
    }
}
