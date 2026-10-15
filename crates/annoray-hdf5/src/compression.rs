//! Compression codecs beyond deflate: HDF5 filter plugins.

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
