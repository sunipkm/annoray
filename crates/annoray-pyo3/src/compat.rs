//! Finding the `annoray` Python package, once, and checking that it can exchange values with
//! this crate.

use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;

/// The version of the exchange format ([`parts`](crate::parts)). The `annoray` package and
/// this crate exchange values only when both name the same one.
pub const PROTOCOL: u32 = 1;

/// This crate's version, which the `annoray` package's must be compatible with.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `annoray._annoray`, once checked.
static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// The leading `X.Y.Z` of a Cargo (`0.3.0-pre1`) or Python (`0.3.0rc1`, `0.3.1.dev4`) version.
fn release(version: &str) -> Option<[u64; 3]> {
    let mut parts = [0u64; 3];
    let mut rest = version;
    for slot in &mut parts {
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        let (n, tail) = rest.split_at(digits);
        *slot = n.parse().ok()?;
        rest = tail.strip_prefix('.').unwrap_or("");
    }
    Some(parts)
}

/// Whether the package at `python` can serve this crate at `rust`: Cargo's caret rule (the
/// first non-zero part of `X.Y.Z` agrees, and all parts before it), on the release numbers.
#[must_use]
pub fn compatible(rust: &str, python: &str) -> bool {
    match (release(rust), release(python)) {
        (Some([0, 0, z]), Some([0, 0, w])) => z == w,
        (Some([0, y, _]), Some([0, v, _])) => y == v,
        (Some([x, _, _]), Some([u, _, _])) => x == u,
        _ => false,
    }
}

fn check(py: Python<'_>) -> PyResult<Py<PyModule>> {
    let package = py.import("annoray").map_err(|e| {
        PyImportError::new_err(format!(
            "annoray-pyo3 {VERSION} needs the annoray Python package ({e})"
        ))
    })?;
    let version: String = package.getattr("__version__")?.extract()?;
    if !compatible(VERSION, &version) {
        return Err(PyImportError::new_err(format!(
            "annoray-pyo3 {VERSION} cannot exchange values with annoray {version}: install a \
             compatible annoray (`pip install \"annoray=={VERSION}\"`), or build against the \
             annoray-pyo3 that matches it"
        )));
    }
    let module = py.import("annoray._annoray")?;
    let protocol: u32 = match module.getattr("_interop_protocol") {
        Ok(f) => f.call0()?.extract()?,
        Err(_) => 0,
    };
    if protocol != PROTOCOL {
        return Err(PyImportError::new_err(format!(
            "annoray {version} exchanges values with protocol {protocol}, annoray-pyo3 \
             {VERSION} with protocol {PROTOCOL}: install matching versions"
        )));
    }
    Ok(module.unbind())
}

/// `annoray._annoray`, after checking (once per process) that the installed `annoray` is
/// compatible with this crate. Every conversion calls it; calling it directly makes an
/// extension fail early, at import, rather than at its first conversion.
///
/// # Errors
/// `ImportError` if `annoray` is not installed, or its version or exchange protocol does not
/// match this crate's. The check runs again on the next call after a failure.
pub fn annoray_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    Ok(MODULE.get_or_try_init(py, || check(py))?.bind(py))
}
