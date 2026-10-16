//! A local directory store: keys are `/`-separated paths below the root.

use std::path::{Path, PathBuf};

use crate::Error;

/// A Zarr store in a local directory.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

fn io(e: &std::io::Error, path: &Path) -> Error {
    Error::Io(format!("{}: {e}", path.display()))
}

impl Store {
    /// The store rooted at `root` (which need not exist yet).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The file for `key`, refusing keys that would leave the store (`..`, absolute parts).
    fn path(&self, key: &str) -> Result<PathBuf, Error> {
        let mut p = self.root.clone();
        for part in key.split('/').filter(|s| !s.is_empty()) {
            if part == ".." || part == "." || part.contains('\\') || Path::new(part).is_absolute() {
                return Err(Error::Invalid(format!("key {key:?} leaves the store")));
            }
            p.push(part);
        }
        Ok(p)
    }

    /// The bytes under `key`, or `None` if there are none.
    ///
    /// # Errors
    /// [`Error::Io`] for failures other than a missing file.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, Error> {
        let p = self.path(key)?;
        match std::fs::read(&p) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(&e, &p)),
        }
    }

    /// Store `bytes` under `key`, creating directories as needed.
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn set(&self, key: &str, bytes: &[u8]) -> Result<(), Error> {
        let p = self.path(key)?;
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).map_err(|e| io(&e, dir))?;
        }
        std::fs::write(&p, bytes).map_err(|e| io(&e, &p))
    }

    /// Remove `key` if present.
    ///
    /// # Errors
    /// [`Error::Io`] for failures other than a missing file.
    pub fn delete(&self, key: &str) -> Result<(), Error> {
        let p = self.path(key)?;
        match std::fs::remove_file(&p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(io(&e, &p)),
        }
    }

    /// Remove everything below `prefix` (a group or array), if present.
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn delete_dir(&self, prefix: &str) -> Result<(), Error> {
        let p = self.path(prefix)?;
        match std::fs::remove_dir_all(&p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(io(&e, &p)),
        }
    }

    /// The names directly below `prefix` that are directories, sorted.
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn children(&self, prefix: &str) -> Result<Vec<String>, Error> {
        let p = self.path(prefix)?;
        let entries = match std::fs::read_dir(&p) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&e, &p)),
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| io(&e, &p))?;
            if entry.file_type().map_err(|e| io(&e, &p))?.is_dir()
                && let Some(name) = entry.file_name().to_str()
            {
                out.push(name.to_owned());
            }
        }
        out.sort();
        Ok(out)
    }
}

/// `prefix/name`, or `name` at the root.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}
