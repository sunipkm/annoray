//! The backend-neutral I/O model.
//!
//! Backends read and write the plain **netCDF data model** ([`RawDataset`]): dimensions,
//! variables with their on-disk element types, attributes exactly as stored, and groups. No CF
//! interpretation happens here; [`cf`](crate::cf) turns a `RawDataset` into a `Dataset` and
//! back.

use std::path::Path;

use indexmap::IndexMap;
use std::borrow::Cow;
use std::sync::Arc;

use crate::{Attrs, CoreError, DType, LazySource, Name, Owned, TypedArray, View};

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
    /// A compression codec other than deflate.
    pub compression: Option<crate::Compression>,
}

/// A variable's values: in memory (owned, or borrowed from the dataset being written), or a
/// source to read them from on demand.
#[derive(Clone)]
pub enum RawData<'a> {
    /// Values in memory.
    Loaded(TypedArray<Owned>),
    /// Values borrowed from memory (encoding for writing passes unchanged values through
    /// without copying them).
    Borrowed(TypedArray<View<'a>>),
    /// Values read on demand (only the parts selected).
    Lazy(Arc<dyn LazySource>),
}

impl std::fmt::Debug for RawData<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded(a) => std::fmt::Debug::fmt(a, f),
            Self::Borrowed(a) => write!(f, "Borrowed({a:?})"),
            Self::Lazy(s) => write!(f, "Lazy({s:?})"),
        }
    }
}

impl From<TypedArray<Owned>> for RawData<'_> {
    fn from(a: TypedArray<Owned>) -> Self {
        Self::Loaded(a)
    }
}

impl<'a> From<TypedArray<View<'a>>> for RawData<'a> {
    fn from(a: TypedArray<View<'a>>) -> Self {
        Self::Borrowed(a)
    }
}

impl RawData<'_> {
    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Loaded(a) => a.dtype(),
            Self::Borrowed(a) => a.dtype(),
            Self::Lazy(s) => s.dtype(),
        }
    }

    /// Shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Loaded(a) => a.shape(),
            Self::Borrowed(a) => a.shape(),
            Self::Lazy(s) => s.shape(),
        }
    }

    /// The values in memory, as a view (`None` for a lazy source).
    #[must_use]
    pub fn in_memory(&self) -> Option<TypedArray<View<'_>>> {
        match self {
            Self::Loaded(a) => Some(a.view()),
            Self::Borrowed(a) => Some(a.view()),
            Self::Lazy(_) => None,
        }
    }

    /// The values in the box `[start, start + count)`, copying only that part.
    ///
    /// # Errors
    /// [`IoError::Core`] if a lazy source cannot be read.
    pub fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        match self {
            Self::Lazy(s) => Ok(s.read_slice(start, count)?),
            Self::Loaded(a) => Ok(slice_of(&a.view(), start, count)),
            Self::Borrowed(a) => Ok(slice_of(a, start, count)),
        }
    }

    /// The values (read in full if lazy, copied if borrowed).
    ///
    /// # Errors
    /// [`IoError::Core`] if a lazy source cannot be read.
    pub fn values(&self) -> Result<Cow<'_, TypedArray<Owned>>> {
        match self {
            Self::Loaded(a) => Ok(Cow::Borrowed(a)),
            Self::Borrowed(a) => Ok(Cow::Owned(a.to_owned())),
            Self::Lazy(s) => {
                let start = vec![0; s.shape().len()];
                Ok(Cow::Owned(s.read_slice(&start, s.shape())?))
            }
        }
    }

    /// The same values, owning them (borrowed values are copied), so they may live as long
    /// as any `'b`.
    #[must_use]
    pub fn into_owned<'b>(self) -> RawData<'b> {
        match self {
            Self::Loaded(a) => RawData::Loaded(a),
            Self::Borrowed(a) => RawData::Loaded(a.to_owned()),
            Self::Lazy(s) => RawData::Lazy(s),
        }
    }
}

/// `a[start .. start + count]`, copied out of a view (never the whole array).
pub(crate) fn slice_of(
    a: &TypedArray<View<'_>>,
    start: &[usize],
    count: &[usize],
) -> TypedArray<Owned> {
    let range = |i: usize| {
        let s = start.get(i).copied().unwrap_or(0);
        ndarray::Slice::from(s..s.saturating_add(count.get(i).copied().unwrap_or(0)))
    };
    crate::array::typed_map!(a, x => x.slice_each_axis(|ax| range(ax.axis.index())).to_owned())
}

/// Values already in memory, served slice by slice: a writer that streams slices takes them
/// from here without first copying the whole array.
#[derive(Debug)]
pub(crate) struct Held(pub(crate) TypedArray<Owned>);

impl LazySource for Held {
    fn dtype(&self) -> DType {
        self.0.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> crate::Result<TypedArray<Owned>> {
        Ok(slice_of(&self.0.view(), start, count))
    }
}

/// A variable as stored: raw values in the on-disk type, attributes verbatim.
#[derive(Debug, Clone)]
pub struct RawVariable<'a> {
    /// Dimension names.
    pub dims: Vec<Name>,
    /// Values. `NC_CHAR` data is `U8` with [`is_char`](Self::is_char) set.
    pub data: RawData<'a>,
    /// Whether the on-disk type is `NC_CHAR` (text as fixed-width byte arrays).
    pub is_char: bool,
    /// Attributes exactly as stored, including `_FillValue`, `scale_factor`, `units`, ….
    pub attrs: Attrs,
    /// Chunking and compression.
    pub storage: Storage,
}

/// A group (or the root) as stored. `'a` is how long borrowed values
/// ([`RawData::Borrowed`]) live: `'static` for what backends read.
#[derive(Debug, Clone, Default)]
pub struct RawDataset<'a> {
    /// Dimensions defined in this group.
    pub dims: IndexMap<Name, RawDim>,
    /// Variables, in file order.
    pub vars: IndexMap<Name, RawVariable<'a>>,
    /// Group attributes.
    pub attrs: Attrs,
    /// Child groups.
    pub groups: IndexMap<Name, RawDataset<'a>>,
    /// Variables the backend could not represent, with the reason (e.g. compound types).
    pub skipped: Vec<(Name, String)>,
}

impl RawDataset<'_> {
    /// The same dataset, owning all its values (borrowed values are copied), so it may live
    /// as long as any `'b`.
    #[must_use]
    pub fn into_owned<'b>(self) -> RawDataset<'b> {
        RawDataset {
            dims: self.dims,
            vars: self
                .vars
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        RawVariable {
                            dims: v.dims,
                            data: v.data.into_owned(),
                            is_char: v.is_char,
                            attrs: v.attrs,
                            storage: v.storage,
                        },
                    )
                })
                .collect(),
            attrs: self.attrs,
            groups: self
                .groups
                .into_iter()
                .map(|(k, g)| (k, g.into_owned()))
                .collect(),
            skipped: self.skipped,
        }
    }

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
    fn read(&self, path: &Path) -> Result<RawDataset<'static>>;

    /// Open a file lazily: metadata now, variable data as [`RawData::Lazy`] sources that read
    /// only the slices asked for. Backends without lazy support read everything.
    ///
    /// # Errors
    /// [`IoError::Backend`] for unreadable files.
    fn read_lazy(&self, path: &Path) -> Result<RawDataset<'static>> {
        self.read(path)
    }

    /// Write a dataset (and its groups), replacing any existing file.
    ///
    /// # Errors
    /// [`IoError::Format`] if the dataset needs features the format lacks.
    fn write(&self, ds: &RawDataset<'_>, path: &Path, options: &WriteOptions) -> Result<()>;

    /// As [`write`](Self::write), taking the dataset: backends that stream values may then
    /// write loaded arrays without a second in-memory copy.
    ///
    /// # Errors
    /// As for [`write`](Self::write).
    fn write_owned(&self, ds: RawDataset<'_>, path: &Path, options: &WriteOptions) -> Result<()> {
        self.write(&ds, path, options)
    }
}

pub use annoray_hdf5::FileBytes;

impl From<annoray_hdf5::Error> for IoError {
    fn from(e: annoray_hdf5::Error) -> Self {
        match e {
            annoray_hdf5::Error::Unsupported(m) => Self::Unsupported(m),
            annoray_hdf5::Error::NotFound(p) => Self::NoSuchGroup(p),
            annoray_hdf5::Error::Io(m) => Self::Backend {
                backend: "io",
                message: m,
            },
            other => Self::Backend {
                backend: "hdf5",
                message: match other {
                    annoray_hdf5::Error::Invalid(m) => m,
                    other => other.to_string(),
                },
            },
        }
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
    Ok(annoray_hdf5::map_file(path)?)
}

/// Read a whole file into memory (no mapping).
///
/// # Errors
/// [`IoError::Backend`] if the file cannot be read.
pub fn read_file(path: &Path) -> Result<FileBytes> {
    Ok(annoray_hdf5::read_file(path)?)
}
