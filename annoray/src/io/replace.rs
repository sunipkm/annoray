//! Writing a file by replacing it: the new contents go to a temporary file beside it, which
//! is then renamed over the old one.
//!
//! Datasets opened from the old file map it into memory. Truncating it in place would pull the
//! data from under them (a crash, `SIGBUS`, on Linux and macOS when they are next read; an
//! error on Windows), and writing a dataset back to the file it was read from would read from
//! the file being written. A rename leaves the old contents to whatever still has them open,
//! and no reader ever sees a partly written file.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::raw::IoError;

/// Distinguishes the temporary files of concurrent writes in one process.
static NEXT: AtomicU64 = AtomicU64::new(0);

fn failed(backend: &'static str, path: &Path, what: impl std::fmt::Display) -> IoError {
    IoError::Backend {
        backend,
        message: format!("{}: {what}", path.display()),
    }
}

/// The file a write to `path` replaces: `path` itself, or the file a symbolic link at `path`
/// points to (the link is kept, as writing through it would keep it).
fn target(path: &Path, backend: &'static str) -> Result<PathBuf, IoError> {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => {
            std::fs::canonicalize(path).map_err(|e| failed(backend, path, e))
        }
        _ => Ok(path.to_path_buf()),
    }
}

/// A new, empty temporary file beside `target`.
fn temporary(target: &Path, backend: &'static str) -> Result<(PathBuf, File), IoError> {
    let name = target
        .file_name()
        .ok_or_else(|| failed(backend, target, "not a file name"))?
        .to_string_lossy();
    let dir = target
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let pid = std::process::id();
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".{name}.{pid}.{n}.tmp"));
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(f) => return Ok((tmp, f)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) => return Err(failed(backend, target, e)),
        }
    }
}

/// Write `path` through `write`, replacing any file there only once `write` has succeeded.
/// The contents are flushed to the operating system, not synced to disk, as netCDF-C's
/// `nc_close` leaves them. A file that is replaced keeps its permissions.
///
/// # Errors
/// What `write` returns, or [`IoError::Backend`] if the file cannot be created or put in place;
/// either way any file already at `path` is left as it was.
pub(crate) fn write_replacing(
    path: &Path,
    backend: &'static str,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), IoError>,
) -> Result<(), IoError> {
    let target = target(path, backend)?;
    let (tmp, file) = temporary(&target, backend)?;
    let written = (|| {
        let mut out = BufWriter::new(file);
        write(&mut out)?;
        // Closed before the rename: Windows renames only closed files.
        drop(
            out.into_inner()
                .map_err(|e| failed(backend, path, e.error()))?,
        );
        if let Ok(old) = std::fs::metadata(&target) {
            // Best effort: a file system without permissions keeps its default.
            let _ = std::fs::set_permissions(&tmp, old.permissions());
        }
        std::fs::rename(&tmp, &target).map_err(|e| {
            failed(
                backend,
                path,
                format!(
                    "cannot replace the file: {e} (on Windows, close the datasets opened from \
                     it first)"
                ),
            )
        })
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::io::Write;

    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("annoray-replace-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn leftovers(d: &Path) -> Vec<String> {
        std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect()
    }

    #[test]
    fn replaces_and_cleans_up() {
        let d = dir("ok");
        let p = d.join("a.bin");
        std::fs::write(&p, b"old contents").unwrap();
        write_replacing(&p, "test", |out| {
            out.write_all(b"new").unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"new");
        assert!(leftovers(&d).is_empty());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_write_leaves_the_old_file() {
        let d = dir("fail");
        let p = d.join("a.bin");
        std::fs::write(&p, b"old").unwrap();
        let r = write_replacing(&p, "test", |out| {
            out.write_all(b"partial").unwrap();
            Err(IoError::Unsupported("stop".into()))
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read(&p).unwrap(), b"old");
        assert!(leftovers(&d).is_empty());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn writes_through_a_symlink_and_keeps_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let d = dir("link");
        let real = d.join("real.bin");
        std::fs::write(&real, b"old").unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o640)).unwrap();
        let link = d.join("link.bin");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_replacing(&link, "test", |out| {
            out.write_all(b"new").unwrap();
            Ok(())
        })
        .unwrap();
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read(&real).unwrap(), b"new");
        let mode = std::fs::metadata(&real).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
