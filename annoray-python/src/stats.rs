//! `quantile`, `idxmin`/`idxmax`, `weighted` and `coarsen` for data arrays and datasets.

use annoray::{
    Boundary, Coarsen, CoarsenOptions, DataArray, Dataset, DatasetCoarsen, DatasetWeighted,
    QuantileMethod, ReduceOptions, Reduction, Side, Weighted,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt};

use crate::convert::OrPy;
use crate::native::{
    PyDataArray, PyDataset, merge_kwargs, new_da, new_ds, reduce_dims, reduction_from,
};

/// Probabilities: one number (a scalar quantile) or a sequence.
fn probabilities(q: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, bool)> {
    if q.is_instance_of::<PyFloat>() || q.is_instance_of::<PyInt>() {
        return Ok((vec![q.extract()?], true));
    }
    if let Ok(arr) = q.call_method0("tolist")
        && let Ok(x) = arr.extract::<f64>()
    {
        return Ok((vec![x], true));
    }
    let v: Vec<f64> = match q.call_method0("tolist") {
        Ok(list) => list.extract()?,
        Err(_) => q.extract()?,
    };
    Ok((v, false))
}

fn method(m: &str) -> PyResult<QuantileMethod> {
    Ok(match m {
        "linear" => QuantileMethod::Linear,
        "lower" => QuantileMethod::Lower,
        "higher" => QuantileMethod::Higher,
        "nearest" => QuantileMethod::Nearest,
        "midpoint" => QuantileMethod::Midpoint,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be \"linear\", \"lower\", \"higher\", \"nearest\" or \"midpoint\", \
                 not {other:?}"
            )));
        }
    })
}

fn coarsen_options(boundary: &str, side: &str, coord_func: &str) -> PyResult<CoarsenOptions> {
    let boundary = match boundary {
        "exact" => Boundary::Exact,
        "trim" => Boundary::Trim,
        "pad" => Boundary::Pad,
        other => {
            return Err(PyValueError::new_err(format!(
                "boundary must be \"exact\", \"trim\" or \"pad\", not {other:?}"
            )));
        }
    };
    let side = match side {
        "left" => Side::Left,
        "right" => Side::Right,
        other => {
            return Err(PyValueError::new_err(format!(
                "side must be \"left\" or \"right\", not {other:?}"
            )));
        }
    };
    Ok(CoarsenOptions::default()
        .boundary(boundary)
        .side(side)
        .coord_reduction(reduction_from(coord_func)?))
}

fn windows<'py>(
    dim: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<(String, usize)>> {
    let w: Vec<(String, usize)> = merge_kwargs(dim, kwargs)?
        .into_iter()
        .map(|(d, n)| Ok((d, n.extract()?)))
        .collect::<PyResult<_>>()?;
    if w.is_empty() {
        return Err(PyValueError::new_err(
            "coarsen needs at least one dim=window",
        ));
    }
    Ok(w)
}

#[pymethods]
impl PyDataArray {
    /// Quantiles over `dim` (all dimensions by default), as numpy's `quantile` `method`. A
    /// sequence `q` adds a leading `quantile` dimension.
    #[pyo3(signature = (q, dim=None, method="linear", skipna=true))]
    fn quantile(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<&Bound<'_, PyAny>>,
        method: &str,
        skipna: bool,
    ) -> PyResult<Py<PyDataArray>> {
        let (qs, scalar) = probabilities(q)?;
        let m = self::method(method)?;
        let dims = reduce_dims(dim)?;
        let out = py.detach(|| match dims {
            Some(d) => self.inner.quantile(&qs, scalar, d, m, skipna),
            None => self.inner.quantile(&qs, scalar, .., m, skipna),
        });
        new_da(py, out.py_err()?)
    }

    /// The coordinate label of the minimum along `dim` (NaN/NaT for all-NaN lanes).
    #[pyo3(signature = (dim, skipna=true))]
    fn idxmin(&self, py: Python<'_>, dim: &str, skipna: bool) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.idxmin(dim, skipna)).py_err()?)
    }

    /// The coordinate label of the maximum along `dim` (NaN/NaT for all-NaN lanes).
    #[pyo3(signature = (dim, skipna=true))]
    fn idxmax(&self, py: Python<'_>, dim: &str, skipna: bool) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.idxmax(dim, skipna)).py_err()?)
    }

    /// Weighted reductions: `da.weighted(w).mean("lat")`.
    fn weighted(&self, weights: &Bound<'_, PyDataArray>) -> PyResult<PyWeighted> {
        Ok(PyWeighted {
            inner: Weights::Da(Box::new(
                self.inner.weighted(&weights.borrow().inner).py_err()?,
            )),
        })
    }

    /// Block reductions: `da.coarsen(time=4, boundary="trim").mean()`.
    #[pyo3(signature = (dim=None, boundary="exact", side="left", coord_func="mean", **kwargs))]
    fn coarsen(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        boundary: &str,
        side: &str,
        coord_func: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyCoarsen> {
        let opts = coarsen_options(boundary, side, coord_func)?;
        let w = windows(dim, kwargs)?;
        let refs: Vec<(&str, usize)> = w.iter().map(|(d, n)| (d.as_str(), *n)).collect();
        Ok(PyCoarsen {
            inner: Blocks::Da(Box::new(self.inner.coarsen(&refs, &opts).py_err()?)),
        })
    }
}

#[pymethods]
impl PyDataset {
    /// Quantiles of every numeric variable over `dim`.
    #[pyo3(signature = (q, dim=None, method="linear", skipna=true))]
    fn quantile(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<&Bound<'_, PyAny>>,
        method: &str,
        skipna: bool,
    ) -> PyResult<Py<PyDataset>> {
        let (qs, scalar) = probabilities(q)?;
        let m = self::method(method)?;
        let dims = reduce_dims(dim)?;
        let out = py.detach(|| match dims {
            Some(d) => self.inner.quantile(&qs, scalar, d, m, skipna),
            None => self.inner.quantile(&qs, scalar, .., m, skipna),
        });
        new_ds(py, out.py_err()?)
    }

    /// The coordinate label of each variable's minimum along `dim`.
    #[pyo3(signature = (dim, skipna=true))]
    fn idxmin(&self, py: Python<'_>, dim: &str, skipna: bool) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.idxmin(dim, skipna)).py_err()?)
    }

    /// The coordinate label of each variable's maximum along `dim`.
    #[pyo3(signature = (dim, skipna=true))]
    fn idxmax(&self, py: Python<'_>, dim: &str, skipna: bool) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.idxmax(dim, skipna)).py_err()?)
    }

    /// Weighted reductions of every numeric variable: `ds.weighted(w).mean("lat")`.
    fn weighted(&self, weights: &Bound<'_, PyDataArray>) -> PyResult<PyWeighted> {
        Ok(PyWeighted {
            inner: Weights::Ds(Box::new(
                self.inner.weighted(&weights.borrow().inner).py_err()?,
            )),
        })
    }

    /// Block reductions of every variable: `ds.coarsen(time=4).mean()`.
    #[pyo3(signature = (dim=None, boundary="exact", side="left", coord_func="mean", **kwargs))]
    fn coarsen(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        boundary: &str,
        side: &str,
        coord_func: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyCoarsen> {
        let opts = coarsen_options(boundary, side, coord_func)?;
        let w = windows(dim, kwargs)?;
        let refs: Vec<(&str, usize)> = w.iter().map(|(d, n)| (d.as_str(), *n)).collect();
        Ok(PyCoarsen {
            inner: Blocks::Ds(Box::new(self.inner.coarsen(&refs, &opts).py_err()?)),
        })
    }
}

enum Weights {
    Da(Box<Weighted>),
    Ds(Box<DatasetWeighted>),
}

/// Weighted reductions: from `DataArray.weighted` / `Dataset.weighted`.
#[pyclass(frozen, name = "Weighted", module = "annoray")]
pub(crate) struct PyWeighted {
    inner: Weights,
}

macro_rules! weighted_ops {
    ($($m:ident $doc:literal),*) => {
        #[pymethods]
        impl PyWeighted {$(
            #[doc = $doc]
            #[pyo3(signature = (dim=None))]
            fn $m(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                let dims = reduce_dims(dim)?;
                match &self.inner {
                    Weights::Da(w) => {
                        let out = py.detach(|| match dims {
                            Some(d) => w.$m(d),
                            None => w.$m(..),
                        });
                        Ok(new_da(py, out.py_err()?)?.into_any())
                    }
                    Weights::Ds(w) => {
                        let out = py.detach(|| match dims {
                            Some(d) => w.$m(d),
                            None => w.$m(..),
                        });
                        Ok(new_ds(py, out.py_err()?)?.into_any())
                    }
                }
            }
        )*}
    };
}

weighted_ops!(
    sum_of_weights "The sum of the weights where the data is not NaN.",
    sum "The weighted sum (unit of data × weights).",
    mean "The weighted mean.",
    var "The weighted variance (no degrees-of-freedom correction).",
    std "The weighted standard deviation."
);

#[pymethods]
impl PyWeighted {
    /// Weighted quantiles (xarray's algorithm: Kish's effective sample size, linear).
    #[pyo3(signature = (q, *, dim=None, skipna=true))]
    fn quantile(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<&Bound<'_, PyAny>>,
        skipna: bool,
    ) -> PyResult<Py<PyAny>> {
        let (qs, scalar) = probabilities(q)?;
        let dims = reduce_dims(dim)?;
        match &self.inner {
            Weights::Da(w) => {
                let out = py.detach(|| match dims {
                    Some(d) => w.quantile(&qs, scalar, d, skipna),
                    None => w.quantile(&qs, scalar, .., skipna),
                });
                Ok(new_da(py, out.py_err()?)?.into_any())
            }
            Weights::Ds(w) => {
                let out = py.detach(|| match dims {
                    Some(d) => w.quantile(&qs, scalar, d, skipna),
                    None => w.quantile(&qs, scalar, .., skipna),
                });
                Ok(new_ds(py, out.py_err()?)?.into_any())
            }
        }
    }
}

enum Blocks {
    Da(Box<Coarsen>),
    Ds(Box<DatasetCoarsen>),
}

/// Block reductions: from `coarsen`.
#[pyclass(frozen, name = "Coarsen", module = "annoray")]
pub(crate) struct PyCoarsen {
    inner: Blocks,
}

impl PyCoarsen {
    fn apply(&self, py: Python<'_>, op: Reduction) -> PyResult<Py<PyAny>> {
        let opts = ReduceOptions::default();
        match &self.inner {
            Blocks::Da(c) => {
                let out: DataArray = py.detach(|| c.reduce(op, &opts)).py_err()?;
                Ok(new_da(py, out)?.into_any())
            }
            Blocks::Ds(c) => {
                let out: Dataset = py.detach(|| c.reduce(op, &opts)).py_err()?;
                Ok(new_ds(py, out)?.into_any())
            }
        }
    }
}

#[pymethods]
impl PyCoarsen {
    /// The windows as dimensions: `da.coarsen(x=2).construct(x=("x_coarse", "x_fine"))`.
    #[pyo3(signature = (window_dim=None, **kwargs))]
    fn construct(
        &self,
        py: Python<'_>,
        window_dim: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let names: Vec<(String, String, String)> = merge_kwargs(window_dim, kwargs)?
            .into_iter()
            .map(|(d, pair)| {
                let (outer, inner): (String, String) = pair.extract()?;
                Ok((d, outer, inner))
            })
            .collect::<PyResult<_>>()?;
        let refs: Vec<(&str, &str, &str)> = names
            .iter()
            .map(|(d, o, i)| (d.as_str(), o.as_str(), i.as_str()))
            .collect();
        match &self.inner {
            Blocks::Da(c) => Ok(new_da(py, py.detach(|| c.construct(&refs)).py_err()?)?.into_any()),
            Blocks::Ds(c) => Ok(new_ds(py, py.detach(|| c.construct(&refs)).py_err()?)?.into_any()),
        }
    }
}

macro_rules! coarsen_ops {
    ($($m:ident $op:ident),*) => {
        #[pymethods]
        impl PyCoarsen {$(
            #[doc = concat!("The `", stringify!($m), "` of each window.")]
            fn $m(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                self.apply(py, Reduction::$op)
            }
        )*}
    };
}

coarsen_ops!(
    sum Sum, prod Prod, mean Mean, median Median, std Std, var Var, min Min, max Max,
    count Count
);
