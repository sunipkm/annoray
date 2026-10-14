//! Backend-neutral I/O for xarray-rs.
//!
//! Backends read and write the plain **netCDF data model** ([`RawDataset`]): dimensions,
//! variables with their on-disk element types, attributes exactly as stored, and groups. No CF
//! interpretation happens here; `xr-cf` turns a `RawDataset` into a `Dataset` and back.

use std::ops::Deref;
use std::path::Path;

use indexmap::IndexMap;
use std::borrow::Cow;
use std::sync::Arc;

use xr_core::{Attrs, CoreError, DType, LazySource, Name, Owned, TypedArray};

/// Errors from reading or writing files.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IoError {
    /// The underlying library or file system reported an error.
    #[error("{backend}: {message}")]
    Backend {
        /// Which backend.
        backend: &'static str,
        /// What it said.
        message: String,
    },
    /// The file uses something this backend cannot represent.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The dataset cannot be written in the requested format.
    #[error("cannot write {what} in format {format:?}: {reason}")]
    Format {
        /// The offending item.
        what: String,
        /// The target format.
        format: Format,
        /// Why.
        reason: String,
    },
    /// A requested group does not exist.
    #[error("no group {0:?}")]
    NoSuchGroup(String),
    /// An error from the data model.
    #[error(transparent)]
    Core(#[from] CoreError),
}

/// Result type for I/O.
pub type Result<T, E = IoError> = core::result::Result<T, E>;

/// A file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Format {
    /// netCDF-4 (HDF5 storage), the full data model.
    #[default]
    Netcdf4,
    /// netCDF-4 storage restricted to the classic data model.
    Netcdf4Classic,
    /// netCDF-3 classic (CDF-1).
    Netcdf3Classic,
    /// netCDF-3 with 64-bit offsets (CDF-2).
    Netcdf3Offset64,
    /// netCDF-3 with 64-bit data (CDF-5).
    Netcdf3Data64,
}

impl Format {
    /// Whether this is one of the netCDF-3 formats (no groups, strings, unsigned or 64-bit
    /// integer types, compression or chunking — except CDF-5, which adds the integer types).
    #[must_use]
    pub const fn is_netcdf3(self) -> bool {
        matches!(
            self,
            Self::Netcdf3Classic | Self::Netcdf3Offset64 | Self::Netcdf3Data64
        )
    }

    /// Whether the data model is classic (no groups, no strings).
    #[must_use]
    pub const fn is_classic_model(self) -> bool {
        !matches!(self, Self::Netcdf4)
    }

    /// Whether element type `d` can be stored natively.
    #[must_use]
    pub fn supports(self, d: DType) -> bool {
        match self {
            Self::Netcdf4 => true,
            Self::Netcdf4Classic | Self::Netcdf3Classic | Self::Netcdf3Offset64 => {
                matches!(
                    d,
                    DType::I8 | DType::I16 | DType::I32 | DType::F32 | DType::F64
                )
            }
            Self::Netcdf3Data64 => !matches!(
                d,
                DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
            ),
        }
    }
}

/// A dimension as stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawDim {
    /// Current length.
    pub len: usize,
    /// Whether it is an unlimited (record) dimension.
    pub unlimited: bool,
}

/// How a variable's values are stored, beyond its element type.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Storage {
    /// Chunk shape, if chunked.
    pub chunks: Option<Vec<usize>>,
    /// Deflate level, if compressed.
    pub deflate: Option<u8>,
    /// Whether the shuffle filter is on.
    pub shuffle: bool,
}

/// A variable's values: in memory, or a source to read them from on demand.
#[derive(Clone)]
pub enum RawData {
    /// Values in memory.
    Loaded(TypedArray<Owned>),
    /// Values read on demand (only the parts selected).
    Lazy(Arc<dyn LazySource>),
}

impl std::fmt::Debug for RawData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded(a) => std::fmt::Debug::fmt(a, f),
            Self::Lazy(s) => write!(f, "Lazy({s:?})"),
        }
    }
}

impl From<TypedArray<Owned>> for RawData {
    fn from(a: TypedArray<Owned>) -> Self {
        Self::Loaded(a)
    }
}

impl RawData {
    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Loaded(a) => a.dtype(),
            Self::Lazy(s) => s.dtype(),
        }
    }

    /// Shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Loaded(a) => a.shape(),
            Self::Lazy(s) => s.shape(),
        }
    }

    /// The values (read in full if lazy).
    ///
    /// # Errors
    /// [`IoError::Core`] if a lazy source cannot be read.
    pub fn values(&self) -> Result<Cow<'_, TypedArray<Owned>>> {
        match self {
            Self::Loaded(a) => Ok(Cow::Borrowed(a)),
            Self::Lazy(s) => {
                let start = vec![0; s.shape().len()];
                Ok(Cow::Owned(s.read_slice(&start, s.shape())?))
            }
        }
    }
}

/// A variable as stored: raw values in the on-disk type, attributes verbatim.
#[derive(Debug, Clone)]
pub struct RawVariable {
    /// Dimension names.
    pub dims: Vec<Name>,
    /// Values. `NC_CHAR` data is `U8` with [`is_char`](Self::is_char) set.
    pub data: RawData,
    /// Whether the on-disk type is `NC_CHAR` (text as fixed-width byte arrays).
    pub is_char: bool,
    /// Attributes exactly as stored, including `_FillValue`, `scale_factor`, `units`, ….
    pub attrs: Attrs,
    /// Chunking and compression.
    pub storage: Storage,
}

/// A group (or the root) as stored.
#[derive(Debug, Clone, Default)]
pub struct RawDataset {
    /// Dimensions defined in this group.
    pub dims: IndexMap<Name, RawDim>,
    /// Variables, in file order.
    pub vars: IndexMap<Name, RawVariable>,
    /// Group attributes.
    pub attrs: Attrs,
    /// Child groups.
    pub groups: IndexMap<Name, RawDataset>,
    /// Variables the backend could not represent, with the reason (e.g. compound types).
    pub skipped: Vec<(Name, String)>,
}

impl RawDataset {
    /// The group at a `/`-separated path (`""` or `"/"` is the root).
    ///
    /// # Errors
    /// [`IoError::NoSuchGroup`].
    pub fn group(&self, path: &str) -> Result<&Self> {
        let mut g = self;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            g = g
                .groups
                .get(part)
                .ok_or_else(|| IoError::NoSuchGroup(path.to_owned()))?;
        }
        Ok(g)
    }
}

/// Options for writing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriteOptions {
    /// File format.
    pub format: Format,
}

impl WriteOptions {
    /// Options for a given format.
    #[must_use]
    pub fn format(format: Format) -> Self {
        Self { format }
    }
}

/// A file format implementation.
pub trait Backend {
    /// Short name, for error messages.
    fn name(&self) -> &'static str;

    /// Read a whole file, all groups, into memory.
    ///
    /// # Errors
    /// [`IoError::Backend`] for unreadable files.
    fn read(&self, path: &Path) -> Result<RawDataset>;

    /// Open a file lazily: metadata now, variable data as [`RawData::Lazy`] sources that read
    /// only the slices asked for. Backends without lazy support read everything.
    ///
    /// # Errors
    /// [`IoError::Backend`] for unreadable files.
    fn read_lazy(&self, path: &Path) -> Result<RawDataset> {
        self.read(path)
    }

    /// Write a dataset (and its groups), replacing any existing file.
    ///
    /// # Errors
    /// [`IoError::Format`] if the dataset needs features the format lacks.
    fn write(&self, ds: &RawDataset, path: &Path, options: &WriteOptions) -> Result<()>;
}

/// A file's bytes: memory-mapped (the default for reading) or held in memory.
///
/// Mapping makes opening large files cheap: only the pages actually touched (headers, and the
/// variables or slices that are read) are paged in by the operating system.
#[derive(Debug)]
pub enum FileBytes {
    /// Bytes held in memory.
    Owned(Vec<u8>),
    /// A read-only memory map.
    Mapped(memmap2::Mmap),
}

impl Deref for FileBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(m) => m,
        }
    }
}

impl AsRef<[u8]> for FileBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

fn io_err(path: &Path, e: &std::io::Error) -> IoError {
    IoError::Backend {
        backend: "io",
        message: format!("{}: {e}", path.display()),
    }
}

/// Memory-map a file for reading.
///
/// The file must not be truncated or modified by another process while it is mapped: the
/// operating system may then deliver `SIGBUS` (or show changed bytes). Use [`read_file`] when
/// that cannot be ruled out.
///
/// # Errors
/// [`IoError::Backend`] if the file cannot be opened or mapped.
pub fn map_file(path: &Path) -> Result<FileBytes> {
    let file = std::fs::File::open(path).map_err(|e| io_err(path, &e))?;
    let len = file.metadata().map_err(|e| io_err(path, &e))?.len();
    if len == 0 {
        // Some platforms refuse to map empty files.
        return Ok(FileBytes::Owned(Vec::new()));
    }
    // SAFETY: the map is read-only and outlives every slice taken from it (they borrow from
    // `FileBytes`). The remaining hazard — another process truncating or rewriting the file while
    // it is mapped — is outside Rust's control and documented above; `read_file` avoids it.
    #[allow(unsafe_code)]
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| io_err(path, &e))?;
    Ok(FileBytes::Mapped(map))
}

/// Read a whole file into memory (no mapping).
///
/// # Errors
/// [`IoError::Backend`] if the file cannot be read.
pub fn read_file(path: &Path) -> Result<FileBytes> {
    std::fs::read(path)
        .map(FileBytes::Owned)
        .map_err(|e| io_err(path, &e))
}
