//! How variables are stored on disk, kept alongside them so files round-trip (xarray's
//! `.encoding`).

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
}

/// A compression codec other than deflate, as an HDF5 filter plugin (netCDF-4 / HDF5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Compression {
    /// Zstandard (HDF5 filter 32015, as netCDF-C writes it). The level is recorded for
    /// readers; annoray itself compresses at Zstandard's fastest level.
    Zstd {
        /// Compression level.
        level: i32,
    },
    /// LZ4 (HDF5 filter 32004).
    Lz4,
    /// Blosc (HDF5 filter 32001).
    Blosc {
        /// The codec inside Blosc.
        codec: BloscCodec,
        /// Compression level, 0–9.
        level: u8,
        /// The shuffle Blosc applies first.
        shuffle: BloscShuffle,
    },
}

/// The codec inside Blosc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BloscCodec {
    /// BloscLZ (read only).
    BloscLz,
    /// LZ4.
    Lz4,
    /// LZ4HC (LZ4's high-compression mode; read only, written as LZ4).
    Lz4Hc,
    /// zlib.
    Zlib,
    /// Zstandard.
    Zstd,
}

/// The shuffle Blosc applies before compressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BloscShuffle {
    /// None.
    None,
    /// Bytes grouped by their position in each element.
    Byte,
    /// Bits grouped by their position in each element.
    Bit,
}

/// Storage details of a whole dataset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DatasetEncoding {
    /// Dimensions stored as unlimited (record) dimensions.
    pub unlimited_dims: Vec<Name>,
}
