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

/// Storage details of a whole dataset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DatasetEncoding {
    /// Dimensions stored as unlimited (record) dimensions.
    pub unlimited_dims: Vec<Name>,
}
