//! Data arrays, datasets and trees: [`ArDataArray`], [`ArDataset`], [`ArDataTree`], and the
//! borrowed [`ArDataArrayRef`] for reading values in place.

use annoray::{DataArray, DataTree, Dataset, Unit, Variable};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compat::annoray_module;
use crate::parts;
use crate::units::{is_annoray, not_a};

/// No values are lent from this side: they are moved or copied.
fn copy<'py>(_: &Variable) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(None)
}

/// `annoray._annoray.<function>(arg)`.
fn call<'py>(
    py: Python<'py>,
    function: &str,
    arg: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    annoray_module(py)?.getattr(function)?.call1((arg,))
}

/// The parts of `obj`, an instance of `annoray.<class>`, from `annoray._annoray.<function>`.
fn parts_of<'py>(
    obj: &Bound<'py, PyAny>,
    class: &str,
    function: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if !is_annoray(obj, class)? {
        return Err(not_a(obj, &format!("an annoray.{class}")));
    }
    call(obj.py(), function, obj)
}

macro_rules! boundary {
    ($(#[$doc:meta])* $name:ident($inner:ty), $class:literal, $into:path, $to:path, $from:path,
     $from_parts:literal, $to_parts:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone)]
        pub struct $name(pub $inner);

        impl<'py> IntoPyObject<'py> for $name {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            /// Moves the values into numpy (no copy for numbers), then into annoray (one copy).
            fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                call(py, $from_parts, &$into(py, self.0)?.into_any())
            }
        }

        impl<'a, 'py> IntoPyObject<'py> for &'a $name {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            /// Copies the values into numpy, then into annoray (two copies of numbers).
            fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                call(py, $from_parts, &$to(py, &self.0, &copy)?.into_any())
            }
        }

        impl<'a, 'py> FromPyObject<'a, 'py> for $name {
            type Error = PyErr;

            /// annoray lends its values (read-only numpy views); they are copied once here.
            fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                let obj = obj.to_owned();
                Ok(Self($from(&parts_of(&obj, $class, $to_parts)?)?))
            }
        }
    };
}

boundary!(
    /// A data array at the boundary with Python: `annoray.DataArray` in Python, with its
    /// dimensions, coordinates, units, time zones, attributes, name and values of every
    /// element type.
    ArDataArray(DataArray), "DataArray", parts::dataarray_into_parts, parts::dataarray_to_parts,
    parts::dataarray_from_parts, "_array_from_parts", "_array_to_parts"
);

boundary!(
    /// A dataset at the boundary with Python: `annoray.Dataset` in Python.
    ArDataset(Dataset), "Dataset", parts::dataset_into_parts, parts::dataset_to_parts,
    parts::dataset_from_parts, "_dataset_from_parts", "_dataset_to_parts"
);

/// A tree of datasets at the boundary with Python: `annoray.DataTree` in Python, with the
/// root's name (`None` unless the tree has one).
#[derive(Debug, Clone)]
pub struct ArDataTree {
    /// The tree.
    pub tree: DataTree,
    /// The root's name.
    pub name: Option<String>,
}

impl<'py> IntoPyObject<'py> for ArDataTree {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// Moves the values into numpy (no copy for numbers), then into annoray (one copy).
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let parts = parts::tree_into_parts(py, self.tree, self.name.as_deref())?;
        call(py, "_tree_from_parts", &parts.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ArDataTree {
    type Error = PyErr;

    /// The subtree at this node, copied twice (trees do not lend their values).
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        let (tree, name) = parts::tree_from_parts(&parts_of(&obj, "DataTree", "_tree_to_parts")?)?;
        Ok(Self { tree, name })
    }
}

/// One variable of an [`ArDataArrayRef`]: its metadata, and its values in place.
#[derive(Debug, Clone)]
pub struct ArVariableRef<'py> {
    parts: Bound<'py, PyDict>,
}

impl<'py> ArVariableRef<'py> {
    fn get(&self, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self.parts
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("annoray parts: missing {key:?}")))
    }

    fn opt(&self, key: &str) -> PyResult<Option<String>> {
        let v = self.get(key)?;
        if v.is_none() {
            Ok(None)
        } else {
            v.extract().map(Some)
        }
    }

    /// The dimensions.
    ///
    /// # Errors
    /// For malformed parts.
    pub fn dims(&self) -> PyResult<Vec<String>> {
        self.get("dims")?.extract()
    }

    /// The unit, if any.
    ///
    /// # Errors
    /// `annoray.UnitError` if it does not parse.
    pub fn unit(&self) -> PyResult<Option<Unit>> {
        let py = self.parts.py();
        self.opt("units")?
            .map(|u| {
                u.parse::<Unit>()
                    .map_err(|e| crate::units::unit_error(py, e))
            })
            .transpose()
    }

    /// The kind of values: `"number"`, `"text"`, `"datetime"`, `"timedelta"` or `"cfdate"`.
    ///
    /// # Errors
    /// For malformed parts.
    pub fn kind(&self) -> PyResult<String> {
        self.get("kind")?.extract()
    }

    /// The values as Python sees them (a read-only numpy view of annoray's data for numbers,
    /// date-times and durations).
    ///
    /// # Errors
    /// For malformed parts.
    pub fn values(&self) -> PyResult<Bound<'py, PyAny>> {
        self.get("values")
    }

    /// The values in place, as `T` (for date-times and durations, `i64` nanoseconds): a
    /// read-only view of annoray's own data, without a copy.
    ///
    /// # Errors
    /// `TypeError` if the values are not of type `T` (or are text or CF dates).
    pub fn view<T: numpy::Element>(&self) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
        let values = self.values()?;
        let values = match self.kind()?.as_str() {
            "datetime" | "timedelta" => values.call_method1("view", ("int64",))?,
            "number" => values,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "{other} values have no numeric view"
                )));
            }
        };
        let found = values.getattr("dtype")?.str()?.to_string();
        let arr = values.cast_into::<PyArrayDyn<T>>().map_err(|_| {
            PyTypeError::new_err(format!(
                "the values are {found}, not {}",
                std::any::type_name::<T>()
            ))
        })?;
        Ok(arr.readonly())
    }

    /// The variable, copied into Rust.
    ///
    /// # Errors
    /// As for [`parts::variable_from_parts`].
    pub fn to_variable(&self) -> PyResult<Variable> {
        parts::variable_from_parts(self.parts.as_any())
    }
}

/// An `annoray.DataArray` borrowed from Python: its values read in place, without a copy
/// (see [`ArVariableRef::view`]), for as long as the GIL is held.
#[derive(Debug, Clone)]
pub struct ArDataArrayRef<'py> {
    parts: Bound<'py, PyDict>,
}

impl<'py> ArDataArrayRef<'py> {
    fn variable_at(&self, d: &Bound<'py, PyAny>) -> PyResult<ArVariableRef<'py>> {
        Ok(ArVariableRef {
            parts: d.cast::<PyDict>()?.clone(),
        })
    }

    /// The name, if any.
    ///
    /// # Errors
    /// For malformed parts.
    pub fn name(&self) -> PyResult<Option<String>> {
        match self.parts.get_item("name")? {
            Some(n) if !n.is_none() => n.extract().map(Some),
            _ => Ok(None),
        }
    }

    /// The data variable.
    ///
    /// # Errors
    /// For malformed parts.
    pub fn variable(&self) -> PyResult<ArVariableRef<'py>> {
        let v = self
            .parts
            .get_item("variable")?
            .ok_or_else(|| PyValueError::new_err("annoray parts: missing \"variable\""))?;
        self.variable_at(&v)
    }

    /// The coordinates, by name, in order.
    ///
    /// # Errors
    /// For malformed parts.
    pub fn coords(&self) -> PyResult<Vec<(String, ArVariableRef<'py>)>> {
        let c = self
            .parts
            .get_item("coords")?
            .ok_or_else(|| PyValueError::new_err("annoray parts: missing \"coords\""))?;
        c.cast::<PyDict>()?
            .iter()
            .map(|(k, v)| Ok((k.extract()?, self.variable_at(&v)?)))
            .collect()
    }

    /// The data array, copied into Rust.
    ///
    /// # Errors
    /// As for [`parts::dataarray_from_parts`].
    pub fn to_dataarray(&self) -> PyResult<DataArray> {
        parts::dataarray_from_parts(self.parts.as_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ArDataArrayRef<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        let parts = parts_of(&obj, "DataArray", "_array_to_parts")?;
        Ok(Self {
            parts: parts.cast_into::<PyDict>()?,
        })
    }
}
