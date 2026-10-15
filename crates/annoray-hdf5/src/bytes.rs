//! A file's bytes, memory-mapped or in memory.

use std::ops::Deref;
use std::path::Path;

use crate::{Error, Result};

/// A file's bytes: memory-mapped (the default for reading) or held in memory.
///
/// Mapping makes opening large files cheap: only the pages actually touched (headers, and the
/// datasets or slices that are read) are paged in by the operating system.
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

fn io_err(path: &Path, e: &std::io::Error) -> Error {
    Error::Io(format!("{}: {e}", path.display()))
}

/// Memory-map a file for reading.
///
/// The file must not be truncated or modified by another process while it is mapped: the
/// operating system may then deliver `SIGBUS` (or show changed bytes). Use [`read_file`] when
/// that cannot be ruled out.
///
/// # Errors
/// [`Error::Io`] if the file cannot be opened or mapped.
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
/// [`Error::Io`] if the file cannot be read.
pub fn read_file(path: &Path) -> Result<FileBytes> {
    std::fs::read(path)
        .map(FileBytes::Owned)
        .map_err(|e| io_err(path, &e))
}
