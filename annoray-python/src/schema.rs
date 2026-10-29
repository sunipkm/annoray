//! `annoray.Schema`: what a dataset must hold, checked with every problem reported.

use annoray::schema::{Schema, Var};
use annoray::{AttrValue, DType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyEllipsis, PyFrozenSet, PyList, PySet};

use crate::convert::{attr_from_py, parse_unit};
use crate::native::PyDataset;
use crate::units::PyDimension;

pyo3::create_exception!(
    annoray,
    SchemaError,
    PyValueError,
    "A dataset does not match a `Schema`; `problems` lists every way it differs."
);

/// The keys of a variable's dict.
const VAR_KEYS: &[&str] = &[
    "dims",
    "ndim",
    "dtype",
    "units",
    "convertible",
    "dimension",
    "sizes",
    "min_sizes",
    "attrs",
    "optional",
];

/// What a dataset must hold. `data_vars` and `coords` map names to dicts of what each
/// variable must be (each key optional):
///
/// - `dims`: its dimensions, in order; `ndim`: only how many;
/// - `dtype`: its element type (a name or numpy dtype: `"float32"`, `np.int64`, …);
/// - `units`: its unit (equal as units compare: `"km"` is `"1000 m"`), or `None` for no
///   unit; `convertible`: a unit it must convert to; `dimension`: a `Dimension`, or a unit
///   whose dimension it must have (`"m"`: any length);
/// - `sizes` / `min_sizes`: dimension lengths, exactly or at least;
/// - `attrs`: its attributes, as for the dataset's;
/// - `optional`: whether it may be absent (default `False`).
///
/// `attrs` maps attribute names to the value they must have, a `set` of values they may have,
/// or `...` for any value. Variables not named are allowed unless `closed`.
#[pyclass(name = "Schema", module = "annoray", frozen)]
pub(crate) struct PySchema {
    inner: Schema,
    shown: String,
}

fn dtype_of(v: &Bound<'_, PyAny>) -> PyResult<DType> {
    let name: String = match v.extract::<String>() {
        Ok(s) => s,
        Err(_) => v
            .py()
            .import("numpy")?
            .call_method1("dtype", (v,))?
            .getattr("name")?
            .extract()?,
    };
    if let Some(d) = crate::raw::dtype_from_name(&name) {
        return Ok(d);
    }
    match name.as_str() {
        n if n.starts_with("datetime64") => Ok(DType::DateTime),
        n if n.starts_with("timedelta64") => Ok(DType::TimeDelta),
        "cfdate" | "cftime" => Ok(DType::CfDate),
        other => Err(PyValueError::new_err(format!(
            "Schema: unknown dtype {other:?}"
        ))),
    }
}

/// An attribute rule: a value, a set of values, or `...`.
enum AttrRule {
    Present,
    Eq(AttrValue),
    In(Vec<AttrValue>),
}

fn attr_rule(v: &Bound<'_, PyAny>) -> PyResult<AttrRule> {
    if v.is_instance_of::<PyEllipsis>() {
        return Ok(AttrRule::Present);
    }
    if v.is_instance_of::<PySet>() || v.is_instance_of::<PyFrozenSet>() {
        let mut values = Vec::new();
        for item in v.try_iter()? {
            values.push(attr_from_py(&item?)?);
        }
        return Ok(AttrRule::In(values));
    }
    Ok(AttrRule::Eq(attr_from_py(v)?))
}

fn attr_rules(d: Option<Bound<'_, PyAny>>) -> PyResult<Vec<(String, AttrRule)>> {
    let Some(d) = d.filter(|d| !d.is_none()) else {
        return Ok(Vec::new());
    };
    let d = d
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("Schema: attrs must be a dict"))?;
    d.iter()
        .map(|(k, v)| Ok((k.extract()?, attr_rule(&v)?)))
        .collect()
}

fn sizes(d: Option<Bound<'_, PyAny>>) -> PyResult<Vec<(String, usize)>> {
    match d.filter(|d| !d.is_none()) {
        None => Ok(Vec::new()),
        Some(d) => d
            .extract::<std::collections::BTreeMap<String, usize>>()
            .map(|m| m.into_iter().collect()),
    }
}

/// A variable's dict as a `Var`, and whether it is optional.
fn var_of(name: &str, spec: &Bound<'_, PyAny>) -> PyResult<(Var, bool)> {
    let d = spec.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!("Schema: the entry of {name:?} must be a dict"))
    })?;
    for (k, _) in d.iter() {
        let k: String = k.extract()?;
        if !VAR_KEYS.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Schema: {name:?} has unknown key {k:?} (expected one of {VAR_KEYS:?})"
            )));
        }
    }
    let get = |k: &str| d.get_item(k);
    let mut v = match get("dims")? {
        Some(dims) if !dims.is_none() => {
            let dims: Vec<String> = if let Ok(one) = dims.extract::<String>() {
                vec![one]
            } else {
                dims.extract()?
            };
            Var::new(dims)
        }
        _ => Var::any(),
    };
    if let Some(n) = get("ndim")?.filter(|n| !n.is_none()) {
        v = v.ndim(n.extract()?);
    }
    if let Some(t) = get("dtype")?.filter(|t| !t.is_none()) {
        v = v.dtype(dtype_of(&t)?);
    }
    let unit_rules = ["units", "convertible", "dimension"]
        .iter()
        .filter(|k| d.contains(**k).unwrap_or(false))
        .count();
    if unit_rules > 1 {
        return Err(PyValueError::new_err(format!(
            "Schema: {name:?} gives more than one of units, convertible and dimension"
        )));
    }
    if let Some(u) = get("units")? {
        v = if u.is_none() {
            v.no_unit()
        } else {
            v.unit(&u.extract::<String>()?)
        };
    }
    if let Some(u) = get("convertible")? {
        v = v.convertible(&u.extract::<String>()?);
    }
    if let Some(dim) = get("dimension")? {
        let dim = match dim.cast::<PyDimension>() {
            Ok(p) => p.get().inner,
            Err(_) => parse_unit(&dim.extract::<String>()?)?.dim(),
        };
        v = v.dimension_of(dim);
    }
    for (dim, n) in sizes(get("sizes")?)? {
        v = v.size(dim, n);
    }
    for (dim, n) in sizes(get("min_sizes")?)? {
        v = v.min_size(dim, n);
    }
    for (k, rule) in attr_rules(get("attrs")?)? {
        v = match rule {
            AttrRule::Present => v.attr(k),
            AttrRule::Eq(a) => v.attr_eq(k, a),
            AttrRule::In(a) => v.attr_in(k, a),
        };
    }
    let optional = match get("optional")? {
        Some(o) => o.is_truthy()?,
        None => false,
    };
    Ok((v, optional))
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (data_vars=None, coords=None, attrs=None, *, closed=false))]
    fn new(
        data_vars: Option<&Bound<'_, PyDict>>,
        coords: Option<&Bound<'_, PyDict>>,
        attrs: Option<&Bound<'_, PyAny>>,
        closed: bool,
    ) -> PyResult<Self> {
        let mut s = Schema::new();
        for (k, rule) in attr_rules(attrs.cloned())? {
            s = match rule {
                AttrRule::Present => s.attr(k),
                AttrRule::Eq(a) => s.attr_eq(k, a),
                AttrRule::In(a) => s.attr_in(k, a),
            };
        }
        for (k, spec) in coords.into_iter().flat_map(|d| d.iter()) {
            let k: String = k.extract()?;
            let (v, optional) = var_of(&k, &spec)?;
            s = if optional {
                s.optional_coord(k, v)
            } else {
                s.coord(k, v)
            };
        }
        for (k, spec) in data_vars.into_iter().flat_map(|d| d.iter()) {
            let k: String = k.extract()?;
            let (v, optional) = var_of(&k, &spec)?;
            s = if optional {
                s.optional(k, v)
            } else {
                s.var(k, v)
            };
        }
        if closed {
            s = s.closed();
        }
        let names = |d: Option<&Bound<'_, PyDict>>| -> PyResult<Vec<String>> {
            d.map_or(Ok(Vec::new()), |d| d.keys().extract())
        };
        let shown = format!(
            "<annoray.Schema data_vars={:?} coords={:?}{}>",
            names(data_vars)?,
            names(coords)?,
            if closed { " closed" } else { "" }
        );
        Ok(Self { inner: s, shown })
    }

    /// Every way `ds` differs from the schema, as messages (empty if it matches). Only
    /// metadata is read: lazy values stay unread.
    fn problems(&self, ds: &Bound<'_, PyDataset>) -> Vec<String> {
        let ds = ds.borrow();
        self.inner
            .problems(&ds.inner)
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Raise `SchemaError` (with `problems`, every way `ds` differs) unless `ds` matches.
    fn check(&self, py: Python<'_>, ds: &Bound<'_, PyDataset>) -> PyResult<()> {
        let ds = ds.borrow();
        match self.inner.check(&ds.inner) {
            Ok(()) => Ok(()),
            Err(e) => {
                let err = SchemaError::new_err(e.to_string());
                let problems: Vec<String> = e.problems.iter().map(ToString::to_string).collect();
                err.value(py)
                    .setattr("problems", PyList::new(py, problems)?)?;
                Err(err)
            }
        }
    }

    fn __repr__(&self) -> String {
        self.shown.clone()
    }
}
