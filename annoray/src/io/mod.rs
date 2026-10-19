//! Reading and writing files.
//!
//! netCDF-3, netCDF-4, HDF5 and Zarr (v2 and v3 directory stores) are read and written in
//! pure Rust; files are memory-mapped.
//! The optional `netcdf-c` feature adds the C libraries as an alternative backend.
//!
//! ```no_run
//! # fn main() -> Result<(), annoray::Error> {
//! use annoray::io::{Format, ToNetcdf, WriteOptions, open_dataset};
//!
//! let ds = open_dataset("air.nc")?;
//! ds.to_netcdf_with("copy.nc", &WriteOptions::format(Format::Netcdf3Offset64))?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;

mod nc_writer;
pub mod netcdf3;
pub mod netcdf4;
#[cfg(feature = "netcdf-c")]
pub mod netcdf_c;
mod raw;
pub mod zarr;

pub use raw::{
    Backend, CompoundField, EnumType, FileBytes, Format, IoError, RawData, RawDataset, RawDim,
    RawVariable, Result, Storage, WriteOptions, map_file, read_file,
};

pub use crate::cf::{DecodeOptions, EncodeOptions};
pub use nc_writer::NetcdfWriter;

use crate::{DataTree, Dataset, DatasetBase, Error, Repr};

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

    /// Read data lazily (the default): opening reads only metadata and dimension coordinates;
    /// other variables are read on first access, and only the part selected. With `false`,
    /// everything is read when opening.
    #[must_use]
    pub fn lazy(mut self, lazy: bool) -> Self {
        self.decode.lazy = lazy;
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
    let raw = if opts.decode.lazy {
        backend.read_lazy(path.as_ref())?
    } else {
        backend.read(path.as_ref())?
    };
    decode_opened(&raw, opts)
}

/// Read a netCDF-3, netCDF-4 or HDF5 file held in memory (its format detected from the
/// bytes); with `lazy`, values are read from the bytes on demand.
///
/// # Errors
/// [`IoError::Unsupported`] for other formats; errors of the format's reader.
pub fn read_bytes(bytes: Vec<u8>, lazy: bool) -> Result<RawDataset<'static>> {
    let bytes = FileBytes::Owned(bytes);
    if netcdf3::is_netcdf3(&bytes) {
        let file = netcdf3::Netcdf3File::from_file_bytes(bytes)?;
        return if lazy {
            Ok(std::sync::Arc::new(file).to_raw_lazy())
        } else {
            file.to_raw()
        };
    }
    if annoray_hdf5::is_hdf5(&bytes) {
        return netcdf4::read_file_bytes(bytes, lazy);
    }
    Err(IoError::Unsupported(
        "the bytes are not a netCDF or HDF5 file".into(),
    ))
}

/// [`open_dataset_with`] for a file held in memory (netCDF-3, netCDF-4 or HDF5).
///
/// # Errors
/// As for [`read_bytes`], and decoding errors.
pub fn open_dataset_from_bytes(bytes: Vec<u8>, opts: &OpenOptions) -> Result<Opened, Error> {
    decode_opened(&read_bytes(bytes, opts.decode.lazy)?, opts)
}

/// [`open_datatree_with`] for a file held in memory (netCDF-3, netCDF-4 or HDF5).
///
/// # Errors
/// As for [`read_bytes`], and decoding errors.
pub fn open_datatree_from_bytes(bytes: Vec<u8>, opts: &OpenOptions) -> Result<OpenedTree, Error> {
    decode_opened_tree(&read_bytes(bytes, opts.decode.lazy)?, opts)
}

fn decode_opened(raw: &RawDataset<'static>, opts: &OpenOptions) -> Result<Opened, Error> {
    let group = raw.group(opts.group.as_deref().unwrap_or(""))?;
    let decoded = crate::cf::decode(group, &opts.decode)?;
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
    let raw = crate::cf::encode(ds, &EncodeOptions::format(opts.format))?;
    backend.write_owned(raw, path.as_ref(), opts)?;
    Ok(())
}

fn not_netcdf(path: &Path) -> IoError {
    IoError::Unsupported(format!("{}: not a netCDF or HDF5 file", path.display()))
}

/// Picks a backend per file, all pure Rust: netCDF-3 via [`netcdf3::Netcdf3`], netCDF-4 and
/// HDF5 via [`netcdf4::Netcdf4`], Zarr directories via [`zarr`]. Files are memory-mapped, so
/// opening is cheap.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoBackend;

impl Backend for AutoBackend {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn read(&self, path: &Path) -> Result<RawDataset<'static>> {
        if zarr::is_zarr(path) {
            return zarr::read(path, false);
        }
        let bytes = map_file(path)?;
        if netcdf3::is_netcdf3(&bytes) {
            return netcdf3::Netcdf3File::from_file_bytes(bytes)?.to_raw();
        }
        if annoray_hdf5::is_hdf5(&bytes) {
            return netcdf4::Netcdf4.read(path);
        }
        Err(not_netcdf(path))
    }

    fn read_lazy(&self, path: &Path) -> Result<RawDataset<'static>> {
        if zarr::is_zarr(path) {
            return zarr::read(path, true);
        }
        let bytes = map_file(path)?;
        if netcdf3::is_netcdf3(&bytes) {
            let file = netcdf3::Netcdf3File::from_file_bytes(bytes)?;
            return Ok(std::sync::Arc::new(file).to_raw_lazy());
        }
        if annoray_hdf5::is_hdf5(&bytes) {
            return netcdf4::Netcdf4.read_lazy(path);
        }
        Err(not_netcdf(path))
    }

    fn write(&self, ds: &RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        if opts.format.is_netcdf3() {
            netcdf3::Netcdf3.write(ds, path, opts)
        } else {
            netcdf4::Netcdf4.write(ds, path, opts)
        }
    }

    fn write_owned(&self, ds: RawDataset, path: &Path, opts: &WriteOptions) -> Result<()> {
        if opts.format.is_netcdf3() {
            netcdf3::Netcdf3.write_owned(ds, path, opts)
        } else {
            netcdf4::Netcdf4.write_owned(ds, path, opts)
        }
    }
}

/// The netCDF-C/HDF5 C-library backend, for comparison with the pure-Rust one.
#[cfg(feature = "netcdf-c")]
#[must_use]
pub fn c_backend() -> &'static dyn Backend {
    &netcdf_c::NetcdfC
}

/// The default backend ([`AutoBackend`]).
#[must_use]
pub fn default_backend() -> &'static dyn Backend {
    &AutoBackend
}

/// Open a netCDF-3, netCDF-4 or HDF5 file and CF-decode it. Data is read lazily: see
/// [`OpenOptions::lazy`] and [`Dataset::load`].
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
    /// Write as netCDF-4 (the default format, as in xarray).
    ///
    /// # Errors
    /// Encoding or I/O errors.
    fn to_netcdf(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.to_netcdf_with(path, &WriteOptions::format(Format::Netcdf4))
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

/// Writing datasets to Zarr stores.
pub trait ToZarr {
    /// Write a Zarr v3 store (as zarr-python 3) with consolidated metadata, replacing `path`.
    ///
    /// # Errors
    /// Encoding or I/O errors.
    fn to_zarr(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.to_zarr_with(path, &zarr::ZarrOptions::default())
    }

    /// Write with options (format, compressor, consolidated metadata).
    ///
    /// # Errors
    /// Encoding or I/O errors.
    fn to_zarr_with(&self, path: impl AsRef<Path>, opts: &zarr::ZarrOptions) -> Result<(), Error>;
}

impl<S: Repr> ToZarr for DatasetBase<S> {
    fn to_zarr_with(&self, path: impl AsRef<Path>, opts: &zarr::ZarrOptions) -> Result<(), Error> {
        let raw = crate::cf::encode(self, &EncodeOptions::format(Format::Netcdf4))?;
        zarr::write(&raw, path.as_ref(), opts)?;
        Ok(())
    }
}

/// Open a Zarr store (v2 or v3) and CF-decode it, lazily. [`open_dataset`] detects stores
/// too; this names the intent.
///
/// # Errors
/// I/O or decoding errors.
pub fn open_zarr(path: impl AsRef<Path>) -> Result<Dataset, Error> {
    open_dataset(path)
}

// ----- data trees -----

/// A data tree opened from a file, with any non-fatal decoding warnings.
#[derive(Debug)]
pub struct OpenedTree {
    /// The tree.
    pub tree: DataTree,
    /// Problems that did not stop decoding, prefixed with their group's path.
    pub warnings: Vec<String>,
}

fn decode_tree(
    raw: &RawDataset,
    opts: &DecodeOptions,
    path: &str,
    warnings: &mut Vec<String>,
) -> Result<DataTree, Error> {
    let decoded = crate::cf::decode(raw, opts)?;
    let shown = if path.is_empty() { "/" } else { path };
    warnings.extend(
        decoded
            .warnings
            .into_iter()
            .map(|w| format!("{shown}: {w}")),
    );
    let mut tree = DataTree::new(decoded.dataset);
    for (name, group) in &raw.groups {
        let child = decode_tree(group, opts, &format!("{path}/{name}"), warnings)?;
        tree.add_child(name.clone(), child)?;
    }
    Ok(tree)
}

/// Open every group of a file as a [`DataTree`] (with a specific backend).
///
/// # Errors
/// I/O or decoding errors, or a group that disagrees with its parent's coordinates.
pub fn open_datatree_with_backend(
    backend: &dyn Backend,
    path: impl AsRef<Path>,
    opts: &OpenOptions,
) -> Result<OpenedTree, Error> {
    let raw = if opts.decode.lazy {
        backend.read_lazy(path.as_ref())?
    } else {
        backend.read(path.as_ref())?
    };
    decode_opened_tree(&raw, opts)
}

fn decode_opened_tree(raw: &RawDataset<'static>, opts: &OpenOptions) -> Result<OpenedTree, Error> {
    let root = opts.group.as_deref().unwrap_or("");
    let mut warnings = Vec::new();
    let tree = decode_tree(raw.group(root)?, &opts.decode, "", &mut warnings)?;
    Ok(OpenedTree { tree, warnings })
}

/// Open every group of a file as a [`DataTree`].
///
/// # Errors
/// As for [`open_datatree_with_backend`].
pub fn open_datatree(path: impl AsRef<Path>) -> Result<DataTree, Error> {
    Ok(open_datatree_with_backend(default_backend(), path, &OpenOptions::default())?.tree)
}

/// Open a data tree with options (the `group` option picks the root), keeping warnings.
///
/// # Errors
/// As for [`open_datatree_with_backend`].
pub fn open_datatree_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<OpenedTree, Error> {
    open_datatree_with_backend(default_backend(), path, opts)
}

fn encode_tree(tree: &DataTree, format: Format) -> Result<RawDataset<'_>, Error> {
    let mut raw = crate::cf::encode(tree.dataset(), &EncodeOptions::format(format))?;
    if !tree.children().is_empty() && format != Format::Netcdf4 {
        return Err(IoError::Format {
            what: "groups".into(),
            format,
            reason: "only netCDF-4 has groups".into(),
        }
        .into());
    }
    for (name, child) in tree.children() {
        raw.groups.insert(name.clone(), encode_tree(child, format)?);
    }
    Ok(raw)
}

impl ToZarr for DataTree {
    fn to_zarr_with(&self, path: impl AsRef<Path>, opts: &zarr::ZarrOptions) -> Result<(), Error> {
        let raw = encode_tree(self, Format::Netcdf4)?;
        zarr::write(&raw, path.as_ref(), opts)?;
        Ok(())
    }
}

impl ToNetcdf for DataTree {
    fn to_netcdf_with(&self, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<(), Error> {
        let raw = encode_tree(self, opts.format)?;
        default_backend().write_owned(raw, path.as_ref(), opts)?;
        Ok(())
    }
}

// ----- many files -----

/// How [`open_mfdataset`] puts files together.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Combine {
    /// Order the files by the one dimension whose coordinates differ, and concatenate along
    /// it (xarray's `combine="by_coords"`); merge if none differs. The default.
    #[default]
    ByCoords,
    /// Concatenate in the given order along this dimension, existing or new (xarray's
    /// `combine="nested"` with one `concat_dim`).
    Nested(String),
}

/// Options for [`open_mfdataset`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MfOptions {
    /// How each file is opened (lazily by default).
    pub open: OpenOptions,
    /// How the files are put together.
    pub combine: Combine,
    /// How to concatenate (xarray's new defaults: exact join, minimal variables).
    pub concat: crate::ConcatOptions,
}

impl MfOptions {
    /// Concatenate in the given order along `dim`.
    #[must_use]
    pub fn concat_dim(mut self, dim: impl Into<String>) -> Self {
        self.combine = Combine::Nested(dim.into());
        self
    }

    /// Options for opening each file.
    #[must_use]
    pub fn open(mut self, open: OpenOptions) -> Self {
        self.open = open;
        self
    }

    /// Options for concatenating.
    #[must_use]
    pub fn concat(mut self, concat: crate::ConcatOptions) -> Self {
        self.concat = concat;
        self
    }
}

/// Open many files as one dataset (xarray's `open_mfdataset`). Files are opened lazily, and
/// the result stays lazy: reading a slice reads only the files (and parts) it covers.
///
/// # Errors
/// [`IoError::Unsupported`] for an empty list; I/O, decoding or combining errors.
pub fn open_mfdataset<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
    opts: &MfOptions,
) -> Result<Opened, Error> {
    let mut warnings = Vec::new();
    let mut datasets = Vec::new();
    for p in paths {
        let opened = open_dataset_with(p.as_ref(), &opts.open)?;
        let shown = p.as_ref().display().to_string();
        warnings.extend(opened.warnings.into_iter().map(|w| format!("{shown}: {w}")));
        datasets.push(opened.dataset);
    }
    if datasets.is_empty() {
        return Err(IoError::Unsupported("open_mfdataset: no files".into()).into());
    }
    let refs: Vec<&Dataset> = datasets.iter().collect();
    let dataset = match &opts.combine {
        Combine::ByCoords => crate::combine_by_coords(&refs, &opts.concat)?,
        Combine::Nested(dim) => crate::concat_datasets(&refs, dim, &opts.concat)?,
    };
    Ok(Opened { dataset, warnings })
}

/// The paths matching a shell pattern (`data/air_*.nc`, `runs/*/out_[0-9].nc`, `**` for any
/// depth), sorted.
///
/// # Errors
/// [`IoError::Backend`] for an invalid pattern or a directory that cannot be read.
pub fn glob(pattern: &str) -> Result<Vec<std::path::PathBuf>, Error> {
    let bad = |message: String| IoError::Backend {
        backend: "glob",
        message,
    };
    let mut out = glob::glob(pattern)
        .map_err(|e| bad(format!("{pattern}: {e}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| bad(e.to_string()))?;
    out.sort();
    Ok(out)
}
