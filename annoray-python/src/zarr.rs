//! `Dataset.to_zarr`: Zarr v2/v3 directory stores, as xarray's `to_zarr` (opening goes
//! through `open_dataset`, which recognises stores).

use std::path::PathBuf;

use annoray::io::ToZarr;
use annoray::io::zarr::{ZarrFormat, ZarrOptions};
use pyo3::exceptions::{PyFileExistsError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::OrPy;
use crate::native::{PyDataset, dict_iter, xr_error};

#[pymethods]
impl PyDataset {
    /// Write a Zarr store. `mode` is `"w-"` (fail if it exists, xarray's default) or `"w"`
    /// (replace); `zarr_format` 2 or 3 (default 3); consolidated metadata unless
    /// `consolidated=False`; `encoding` may set per-variable `chunks` (and the CF encoding
    /// keys `to_netcdf` takes).
    #[pyo3(signature = (store, mode=None, group=None, encoding=None, consolidated=None, zarr_format=None, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn to_zarr(
        &self,
        py: Python<'_>,
        store: PathBuf,
        mode: Option<&str>,
        group: Option<&str>,
        encoding: Option<&Bound<'_, PyDict>>,
        consolidated: Option<bool>,
        zarr_format: Option<u8>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if group.is_some() {
            return Err(PyNotImplementedError::new_err(
                "writing into a group of a store (write a DataTree instead)",
            ));
        }
        if let Some(k) = kwargs.and_then(|k| k.keys().iter().next()) {
            return Err(PyNotImplementedError::new_err(format!(
                "to_zarr option {k}"
            )));
        }
        match mode.unwrap_or("w-") {
            "w-" if store.exists() => {
                return Err(PyFileExistsError::new_err(format!(
                    "{} exists; pass mode=\"w\" to replace it",
                    store.display()
                )));
            }
            "w" | "w-" => {}
            other => {
                return Err(PyNotImplementedError::new_err(format!(
                    "mode={other:?} (appending and region writes are not supported)"
                )));
            }
        }
        let format = match zarr_format.unwrap_or(3) {
            2 => ZarrFormat::V2,
            3 => ZarrFormat::V3,
            other => return Err(PyValueError::new_err(format!("zarr_format={other}"))),
        };
        let opts = ZarrOptions::format(format).consolidated(consolidated.unwrap_or(true));
        let mut ds = self.inner.clone();
        for (k, v) in dict_iter(encoding.map(|e| e.as_any()))? {
            let given = v.cast::<PyDict>()?;
            // zarr's `chunks` is netCDF's `chunksizes`.
            let e = given.copy()?;
            if let Some(c) = e.get_item("chunks")? {
                e.set_item("chunksizes", c)?;
                e.del_item("chunks")?;
            }
            let enc = ds.encoding_of_mut(&k).py_err()?;
            crate::raw::update_encoding(enc, &e)?;
        }
        py.detach(|| ds.to_zarr_with(&store, &opts))
            .map_err(xr_error)
    }
}
