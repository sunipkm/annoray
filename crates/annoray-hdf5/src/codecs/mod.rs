//! HDF5-specific filters: the Bitshuffle filter's framing, scale-offset and N-bit (the general
//! codecs live in `annoray-codecs`). Each treats its input as untrusted: reads are
//! bounds-checked, arithmetic is checked, and output is limited to the chunk's expected size.

pub(crate) mod bitshuffle;
pub(crate) mod nbit;
pub(crate) mod scaleoffset;
pub(crate) mod szip;

use crate::Error;

use crate::bad;

/// An error in a named codec.
pub(crate) fn corrupt(codec: &str) -> Error {
    bad(format!("{codec}: corrupt chunk"))
}

pub(crate) use annoray_codecs::bits::{BitReader, be, le32};
