//! How variables are stored on disk, kept alongside them so files round-trip (xarray's
//! `.encoding`).

use annoray_hdf5::Compression;

use crate::{AttrValue, DType, Name};

/// Storage details of one variable. Filled in when a file is decoded and consulted when it is
/// written; everything is optional, and writers pick sensible defaults for what is unset.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Encoding {
    /// Element type on disk, when it differs from the in-memory type (packed integers,
    /// encoded times, booleans, strings as char arrays).
    pub dtype: Option<DType>,
    /// `_FillValue`, in the on-disk type.
    pub fill_value: Option<AttrValue>,
    /// `missing_value`, in the on-disk type.
    pub missing_value: Option<AttrValue>,
    /// `scale_factor` for packed data.
    pub scale_factor: Option<AttrValue>,
    /// `add_offset` for packed data.
    pub add_offset: Option<AttrValue>,
    /// Signed integers on disk hold unsigned values (`_Unsigned = "true"`).
    pub unsigned: bool,
    /// The `units` string of encoded times (`days since 2000-01-01`).
    pub time_units: Option<String>,
    /// The `calendar` of encoded times.
    pub calendar: Option<String>,
    /// For strings stored as char arrays: the name of the string-length dimension.
    pub char_dim: Option<Name>,
    /// Chunk shape (netCDF-4 / HDF5).
    pub chunks: Option<Vec<usize>>,
    /// Deflate (zlib) level, 1–9 (netCDF-4 / HDF5).
    pub deflate: Option<u8>,
    /// Byte-shuffle filter (netCDF-4 / HDF5).
    pub shuffle: bool,
    /// A compression codec other than deflate (netCDF-4 / HDF5 filter plugins).
    pub compression: Option<Compression>,
    /// Fletcher-32 checksums on chunks (netCDF-4 / HDF5; forces chunking).
    pub fletcher32: bool,
    /// Store numbers big-endian (netCDF-4 / HDF5; netCDF-3 is big-endian anyway).
    pub big_endian: bool,
    /// Store contiguously, unchunked (netCDF-4 / HDF5). Gives way to a chunk shape, filters
    /// or an unlimited dimension (it is often only how a file being rewritten was stored).
    pub contiguous: bool,
    /// Write no `_FillValue` (xarray's `_FillValue: None`); otherwise floats get NaN.
    pub no_fill_value: bool,
    /// netCDF-C's lossy quantization of floats before compression ([`crate::cf::Quantize`]).
    pub quantize: Option<crate::cf::Quantize>,
    /// netCDF4-python's `least_significant_digit` rounding of floats.
    pub least_significant_digit: Option<i32>,
    /// Floats stored as IEEE half precision (`float16`). Read as `f32` (exactly); written back
    /// rounded to nearest, ties to even. HDF5 only: netCDF-C cannot read such variables.
    pub float16: bool,
    /// For the values of a CF contiguous ragged array read from a netCDF-4/HDF5 variable-length
    /// variable: its count variable. Writing the dataset to netCDF-4 rebuilds the
    /// variable-length variable from the two (both must be written).
    pub ragged: Option<Name>,
    /// For a field of a compound (record) variable read from netCDF-4/HDF5: the variable,
    /// member and record type. Writing the dataset to netCDF-4 packs the fields back into one
    /// compound variable when all of them are written.
    pub compound: Option<crate::io::CompoundField>,
    /// An enumeration type (netCDF-4/HDF5): names for the values of an integer variable.
    /// Writing to netCDF-4 defines the type and stores the variable as that type (its base
    /// type the on-disk `dtype`).
    pub enumeration: Option<crate::io::EnumType>,
    /// CF attributes naming other variables (`bounds`, `grid_mapping`, `cell_measures`, …:
    /// [`CF_RELATED`](crate::cf::CF_RELATED)) that `decode_coords_all` took off the variable
    /// when it made those variables coordinates: `(attribute, value)`. Writing puts them back.
    pub cf_related: Vec<(String, String)>,
    /// For a level of a stacked dimension (`stack`): that dimension and the level's
    /// position, which `unstack` reads back.
    pub level_of: Option<(Name, usize)>,
}

/// Compression for every variable of a netCDF-4 / HDF5 file whose own [`Encoding`] sets
/// none (no `deflate`, no `compression`), given when writing
/// ([`WriteOptions::compress`](crate::io::WriteOptions::compress)) instead of in each
/// variable's encoding. Neither the dataset nor its values are changed or copied.
///
/// Variables with fewer than [`min_elements`](Self::min_elements) values (scalars and small
/// coordinates, where filters cost more than they save), strings and variable-length
/// variables, which netCDF-C cannot filter, are written as they are. Chunk shapes come from each
/// variable's encoding, or netCDF-C's defaults. netCDF-3 files have no compression: the
/// option is ignored for them.
///
/// ```
/// use annoray::Compress;
/// use annoray::io::{Format, WriteOptions};
///
/// let opts = WriteOptions::format(Format::Netcdf4).compress(Compress::deflate(4));
/// assert_eq!(opts.compress.map(|c| c.min_elements), Some(256));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Compress {
    /// Deflate (zlib) level, 1–9; `None` when [`codec`](Self::codec) is set.
    pub deflate: Option<u8>,
    /// A codec other than deflate (filter plugins: Zstandard, LZ4, Blosc).
    pub codec: Option<Compression>,
    /// The byte-shuffle filter before compressing (on by default).
    pub shuffle: bool,
    /// The fewest values a variable needs to be compressed (default
    /// [`DEFAULT_MIN_ELEMENTS`](Self::DEFAULT_MIN_ELEMENTS)).
    pub min_elements: usize,
}

impl Compress {
    /// The default [`min_elements`](Self::min_elements).
    pub const DEFAULT_MIN_ELEMENTS: usize = 256;

    /// Deflate at `level` (1–9; clamped), with shuffle.
    #[must_use]
    pub const fn deflate(level: u8) -> Self {
        Self {
            deflate: Some(if level < 1 {
                1
            } else if level > 9 {
                9
            } else {
                level
            }),
            codec: None,
            shuffle: true,
            min_elements: Self::DEFAULT_MIN_ELEMENTS,
        }
    }

    /// Compress with `codec`, with shuffle.
    #[must_use]
    pub const fn codec(codec: Compression) -> Self {
        Self {
            deflate: None,
            codec: Some(codec),
            shuffle: true,
            min_elements: Self::DEFAULT_MIN_ELEMENTS,
        }
    }

    /// With or without the shuffle filter.
    #[must_use]
    pub const fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Compress only variables of at least `n` values.
    #[must_use]
    pub const fn min_elements(mut self, n: usize) -> Self {
        self.min_elements = n;
        self
    }
}

/// Storage details of a whole dataset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DatasetEncoding {
    /// Dimensions stored as unlimited (record) dimensions.
    pub unlimited_dims: Vec<Name>,
}
