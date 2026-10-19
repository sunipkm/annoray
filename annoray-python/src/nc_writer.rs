//! `annoray.NetcdfWriter`: writing into netCDF-4 files in place (streaming a new file
//! from a template, appending along an unlimited dimension, writing regions).

use std::path::PathBuf;
use std::sync::Mutex;

use annoray::Name;
use annoray::io::{NetcdfWriter, WriteOptions};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};

use crate::convert::OrPy;
use crate::native::{PyDataset, dict_iter, new_ds, xr_error};

/// A netCDF-4 file open for writing in place.
///
/// `NetcdfWriter.create(path, template)` makes a new file with the template's dimensions,
/// variables, attributes and encodings, writing its coordinates; its data variables read as
/// fill values until written. `NetcdfWriter(path)` opens an existing file. Then `write(ds,
/// region=...)` writes values into a part of the file and `append(ds, dim)` grows the file
/// along an unlimited dimension; `dataset()` is the file as it now is, opened lazily. Values
/// are encoded with the file's own encoding of each variable. Changes are buffered until
/// `flush()`, `dataset()` or `close()` (also on leaving a `with` block).
#[pyclass(name = "NetcdfWriter", module = "annoray")]
pub(crate) struct PyNetcdfWriter {
    inner: Mutex<Option<NetcdfWriter>>,
}

impl PyNetcdfWriter {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut NetcdfWriter) -> Result<T, annoray::Error>,
    ) -> PyResult<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the writer failed in another thread"))?;
        let w = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("the writer is closed"))?;
        f(w).map_err(xr_error)
    }
}

/// `region`: dimension → start (an int, or a slice whose start is used and whose length must
/// match the data's).
fn region_of(
    region: Option<&Bound<'_, PyDict>>,
    ds: &annoray::Dataset,
) -> PyResult<Vec<(Name, usize)>> {
    let sizes = ds.sizes();
    let mut out = Vec::new();
    for (dim, v) in dict_iter(region.map(|r| r.as_any()))? {
        let len = sizes.get(dim.as_str()).copied().ok_or_else(|| {
            PyValueError::new_err(format!("region dimension {dim:?} is not in the data"))
        })?;
        let start = if let Ok(s) = v.cast::<PySlice>() {
            let start: Option<usize> = s.getattr("start")?.extract()?;
            let stop: Option<usize> = s.getattr("stop")?.extract()?;
            let start = start.unwrap_or(0);
            if let Some(stop) = stop
                && stop.checked_sub(start) != Some(len)
            {
                return Err(PyValueError::new_err(format!(
                    "region {dim:?}: {start}:{stop} for data of length {len}"
                )));
            }
            start
        } else {
            v.extract()?
        };
        out.push((Name::from(dim.as_str()), start));
    }
    Ok(out)
}

#[pymethods]
impl PyNetcdfWriter {
    /// Open an existing netCDF-4 file for writing in place.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let w = py.detach(|| NetcdfWriter::open(&path)).map_err(xr_error)?;
        Ok(Self {
            inner: Mutex::new(Some(w)),
        })
    }

    /// A new netCDF-4 file from `template` (see the class documentation); `encoding` and
    /// `unlimited_dims` as for `to_netcdf`.
    #[staticmethod]
    #[pyo3(signature = (path, template, *, format="NETCDF4", encoding=None, unlimited_dims=None))]
    fn create(
        py: Python<'_>,
        path: PathBuf,
        template: &Bound<'_, PyDataset>,
        format: &str,
        encoding: Option<&Bound<'_, PyDict>>,
        unlimited_dims: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let opts = WriteOptions::format(crate::raw::format_from_name(format)?);
        let this = template.borrow();
        let mut ds = this.inner.view();
        if let Some(u) = unlimited_dims {
            ds.encoding_mut().unlimited_dims = u.into_iter().map(Into::into).collect();
        }
        for (k, v) in dict_iter(encoding.map(|e| e.as_any()))? {
            let given = v.cast::<PyDict>()?;
            crate::raw::check_encoding(
                &k,
                given,
                crate::raw::NETCDF_ENCODING_KEYS,
                Some(opts.format),
            )?;
            let enc = ds.encoding_of_mut(&k).py_err()?;
            crate::raw::update_encoding(enc, given)?;
        }
        let w = py
            .detach(|| NetcdfWriter::create(&path, &ds, &opts))
            .map_err(xr_error)?;
        Ok(Self {
            inner: Mutex::new(Some(w)),
        })
    }

    /// Write `ds` into the file: along each dimension of `region` (dimension → start, or a
    /// slice), from there; along the others, from 0. Coordinates along no region dimension
    /// are not written.
    #[pyo3(signature = (ds, region=None))]
    fn write(
        &self,
        py: Python<'_>,
        ds: &Bound<'_, PyDataset>,
        region: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let ds = ds.borrow();
        let region = region_of(region, &ds.inner)?;
        let data = &ds.inner;
        py.detach(|| self.with(|w| w.write(data, &region)))
    }

    /// Append `ds` along the unlimited dimension `dim`: it must hold every variable of the
    /// file along `dim`.
    fn append(&self, py: Python<'_>, ds: &Bound<'_, PyDataset>, dim: &str) -> PyResult<()> {
        let ds = ds.borrow();
        let data = &ds.inner;
        py.detach(|| self.with(|w| w.append(data, dim)))
    }

    /// The current length of a dimension.
    fn dim_len(&self, dim: &str) -> PyResult<usize> {
        self.with(|w| w.dim_len(dim))
    }

    /// The file as it now is, opened lazily (writes what is buffered first).
    fn dataset(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        let ds = py.detach(|| self.with(NetcdfWriter::dataset))?;
        new_ds(py, ds)
    }

    /// Write what is buffered.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.with(NetcdfWriter::flush))
    }

    /// Write what is buffered and close (further calls fail).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let taken = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the writer failed in another thread"))?
            .take();
        match taken {
            Some(w) => py.detach(|| w.close()).map_err(xr_error),
            None => Ok(()),
        }
    }

    /// Whether `close()` has run.
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the writer failed in another thread"))?
            .is_none())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}
