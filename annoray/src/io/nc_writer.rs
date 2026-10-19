//! Writing into netCDF-4 files in place: a new file made from a template whose data
//! variables are filled in as they are produced (streaming), or an existing file appended
//! to along an unlimited dimension.
//!
//! Values are CF-encoded with the file's own encoding of each variable (packing, fill
//! values, time units, on-disk type), after conversion to the variable's unit in the file.
//! The file is changed through [`annoray_hdf5::Editor`]: see it for the storage that can be
//! changed (netCDF-C's and this crate's files can; HDF5 files written with
//! `libver="latest"` mostly cannot). Only the root group is written to.

use std::path::{Path, PathBuf};

use annoray_hdf5::Editor;
use indexmap::IndexMap;

use super::netcdf4::{self, NON_COORD};
use super::{Backend, Format, IoError, OpenOptions, WriteOptions};
use crate::cf::{DecodeOptions, EncodeOptions};
use crate::units::Unit;
use crate::{DatasetBase, Encoding, Error, Name, Repr, VariableBase};

type Result<T> = std::result::Result<T, Error>;

fn bad(msg: impl Into<String>) -> Error {
    IoError::Unsupported(msg.into()).into()
}

/// A netCDF-4 file open for writing in place (see the module documentation). Changes are
/// buffered and written by [`flush`](Self::flush), [`dataset`](Self::dataset) and
/// [`close`](Self::close); dropping the writer without closing loses what is buffered.
///
/// A new file from a template with no time steps yet, filled in step by step:
///
/// ```
/// use annoray::io::{NetcdfWriter, WriteOptions};
/// use annoray::ndarray::{Array1, Array2};
/// use annoray::{Dataset, Variable};
///
/// fn step(hour: usize, values: Array2<f64>) -> Result<Dataset, Box<dyn std::error::Error>> {
///     let mut ds = Dataset::new();
///     let time = annoray::date_range(&format!("2000-01-01T{hour:02}:00"), values.nrows(), "h")?;
///     ds.add_coord("time", Variable::new(["time"], time.into_dyn())?)?;
///     ds.add_var("temp", Variable::new(["time", "x"], values.into_dyn())?.with_unit("K".parse()?))?;
///     Ok(ds)
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let path = std::env::temp_dir().join(format!("annoray-doc-{}.nc", std::process::id()));
/// let mut template = step(0, Array2::zeros((0, 3)))?;
/// template.add_coord("x", Variable::new(["x"], Array1::from(vec![0.0, 1.0, 2.0]).into_dyn())?)?;
/// template.encoding_mut().unlimited_dims = vec!["time".into()];
/// template.encoding_of_mut("time")?.time_units = Some("hours since 2000-01-01".into());
///
/// let mut w = NetcdfWriter::create(&path, &template, &WriteOptions::default())?;
/// for hour in 0..4 {
///     w.append(&step(hour, Array2::from_elem((1, 3), 280.0 + hour as f64))?, "time")?;
/// }
/// // Readable while it is written.
/// assert_eq!(w.dataset()?.sizes().get("time"), Some(&4));
/// w.close()?;
/// # std::fs::remove_file(&path)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct NetcdfWriter {
    path: PathBuf,
    editor: Editor,
    vars: IndexMap<Name, Var>,
    /// Dimensions without a coordinate variable: their dimension-scale datasets.
    scales: IndexMap<Name, String>,
}

#[derive(Debug)]
struct Var {
    /// The HDF5 dataset.
    h5: String,
    /// Dimensions on disk (a character dimension last for strings stored as characters).
    dims: Vec<Name>,
    encoding: Encoding,
    unit: Option<Unit>,
}

impl NetcdfWriter {
    /// A new netCDF-4 file with `template`'s dimensions, variables, attributes and encodings.
    /// Coordinates are written; data variables are left to [`write`](Self::write) and
    /// [`append`](Self::append) (until then they read as fill values). A dimension of
    /// `template` listed in its encoding's `unlimited_dims` can grow; it may start at 0.
    ///
    /// # Errors
    /// Encoding and I/O errors; [`IoError::Format`] for netCDF-3 formats.
    pub fn create<S: Repr>(
        path: impl AsRef<Path>,
        template: &DatasetBase<S>,
        opts: &WriteOptions,
    ) -> Result<Self> {
        if opts.format.is_netcdf3() {
            return Err(IoError::Format {
                what: "writing in place".into(),
                format: opts.format,
                reason: "only netCDF-4 files can be written in place".into(),
            }
            .into());
        }
        let path = path.as_ref();
        let raw = crate::cf::encode(template, &EncodeOptions::format(opts.format))?;
        let deferred: Vec<Name> = template.data_vars().keys().cloned().collect();
        netcdf4::write_deferred(&raw, path, opts, &deferred)?;
        Self::open(path)
    }

    /// Open an existing netCDF-4 file for writing in place.
    ///
    /// # Errors
    /// I/O errors; [`IoError::Unsupported`] for files this writer cannot change.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let editor = Editor::open(&path).map_err(IoError::from)?;
        let raw = netcdf4::Netcdf4.read_lazy(&path)?;
        let decoded = crate::cf::decode(&raw, &DecodeOptions::default())?.dataset;
        let mut vars = IndexMap::new();
        for (name, v) in &raw.vars {
            let clash = raw.dims.contains_key(name) && v.dims.as_slice() != [name.clone()];
            let h5 = if clash {
                format!("{NON_COORD}{name}")
            } else {
                name.to_string()
            };
            let decoded_var = decoded
                .data_vars()
                .get(name)
                .map(|x| (x.encoding().clone(), x.unit().cloned()))
                .or_else(|| {
                    decoded
                        .coords()
                        .get(name)
                        .map(|x| (x.encoding().clone(), x.unit().cloned()))
                });
            let (encoding, unit) = decoded_var.unwrap_or_default();
            vars.insert(
                name.clone(),
                Var {
                    h5,
                    dims: v.dims.clone(),
                    encoding,
                    unit,
                },
            );
        }
        let scales = raw
            .dims
            .keys()
            .filter(|d| !raw.vars.contains_key(*d))
            .map(|d| (d.clone(), d.to_string()))
            .collect();
        Ok(Self {
            path,
            editor,
            vars,
            scales,
        })
    }

    /// The current length of a dimension (with changes not yet flushed).
    ///
    /// # Errors
    /// [`IoError::Unsupported`] for a dimension the file does not have.
    pub fn dim_len(&mut self, dim: &str) -> Result<usize> {
        let (h5, axis) = self.dim_holder(dim)?;
        let shape = self.editor.shape(&h5).map_err(IoError::from)?;
        Ok(shape.get(axis).copied().unwrap_or(0))
    }

    /// A dataset holding `dim` and its axis: a variable along it, else its scale.
    fn dim_holder(&self, dim: &str) -> Result<(String, usize)> {
        if let Some(h5) = self.scales.get(dim) {
            return Ok((h5.clone(), 0));
        }
        self.vars
            .values()
            .find_map(|v| {
                v.dims
                    .iter()
                    .position(|d| d.as_str() == dim)
                    .map(|k| (v.h5.clone(), k))
            })
            .ok_or_else(|| bad(format!("the file has no dimension {dim:?}")))
    }

    /// Write `ds`'s variables into the file: along each dimension in `region`, starting at
    /// its offset; along the others, from 0. Data variables are written whole; coordinates
    /// only when they lie along a dimension of `region` (others, such as fixed latitudes,
    /// are left as they are). Every variable must be in the file, on the same dimensions.
    ///
    /// # Errors
    /// [`IoError::Unsupported`] for variables or dimensions not in the file, values outside
    /// it, incompatible units or element types; I/O errors.
    pub fn write<S: Repr>(&mut self, ds: &DatasetBase<S>, region: &[(Name, usize)]) -> Result<()> {
        let along = |dims: &[Name]| dims.iter().any(|d| region.iter().any(|(r, _)| r == d));
        // Everything is encoded (and checked) before anything is written.
        let mut ready = Vec::new();
        for (name, c) in ds.coords() {
            if along(c.dims()) {
                ready.push((name.clone(), self.encoded(name, c, region)?));
            }
        }
        for (name, v) in ds.data_vars() {
            ready.push((name.clone(), self.encoded(name, v, region)?));
        }
        for (name, (h5, start, values)) in &ready {
            let shape = self.editor.shape(h5).map_err(IoError::from)?;
            let fits = start
                .iter()
                .zip(values.shape())
                .zip(&shape)
                .all(|((s, c), d)| s.checked_add(*c).is_some_and(|e| e <= *d));
            if !fits {
                return Err(bad(format!(
                    "{name:?}: {:?} values at {start:?} are outside its shape {shape:?}",
                    values.shape()
                )));
            }
        }
        for (name, (h5, start, values)) in ready {
            self.editor
                .write(&h5, &start, &values)
                .map_err(|e| bad(format!("writing {name:?}: {e}")))?;
        }
        Ok(())
    }

    /// Append `ds` along the unlimited dimension `dim` (xarray's `append_dim`): every
    /// variable of the file along `dim` grows by `ds`'s length of it and is written there,
    /// so `ds` must hold all of them; its other variables are not written.
    ///
    /// # Errors
    /// [`IoError::Unsupported`] for a dimension that is not unlimited, a variable along it
    /// that `ds` lacks, and as for [`write`](Self::write).
    pub fn append<S: Repr>(&mut self, ds: &DatasetBase<S>, dim: &str) -> Result<()> {
        let n = ds
            .sizes()
            .get(dim)
            .copied()
            .ok_or_else(|| bad(format!("the data to append has no dimension {dim:?}")))?;
        let (holder, axis) = self.dim_holder(dim)?;
        let max = self.editor.max_shape(&holder).map_err(IoError::from)?;
        if max.get(axis).is_some_and(Option::is_some) {
            return Err(bad(format!(
                "dimension {dim:?} is not unlimited: the file cannot grow along it"
            )));
        }
        let along: Vec<Name> = self
            .vars
            .iter()
            .filter(|(_, v)| v.dims.iter().any(|d| d.as_str() == dim))
            .map(|(k, _)| k.clone())
            .collect();
        let has = |k: &Name| ds.data_vars().contains_key(k) || ds.coords().contains_key(k);
        let missing: Vec<&Name> = along.iter().filter(|k| !has(k)).collect();
        if !missing.is_empty() {
            return Err(bad(format!(
                "appending along {dim:?} needs every variable along it; missing: {missing:?}"
            )));
        }
        let start = self.dim_len(dim)?;
        let grown = start
            .checked_add(n)
            .ok_or_else(|| bad("dimension too long"))?;
        // Everything is encoded before the file changes, so a value that cannot be stored
        // leaves it as it was.
        let region = [(Name::from(dim), start)];
        let mut ready = Vec::new();
        for (name, c) in ds.coords() {
            if c.dims().iter().any(|d| d.as_str() == dim) {
                ready.push((name.clone(), self.encoded(name, c, &region)?));
            }
        }
        for (name, v) in ds.data_vars() {
            if v.dims().iter().any(|d| d.as_str() == dim) {
                ready.push((name.clone(), self.encoded(name, v, &region)?));
            }
        }
        let mut datasets: Vec<String> = along
            .iter()
            .filter_map(|k| self.vars.get(k).map(|v| v.h5.clone()))
            .collect();
        if let Some(h5) = self.scales.get(dim) {
            datasets.push(h5.clone());
        }
        for h5 in datasets {
            let mut shape = self.editor.shape(&h5).map_err(IoError::from)?;
            let k = if self.scales.get(dim) == Some(&h5) {
                0
            } else {
                self.vars
                    .values()
                    .find(|v| v.h5 == h5)
                    .and_then(|v| v.dims.iter().position(|d| d.as_str() == dim))
                    .unwrap_or(0)
            };
            if let Some(s) = shape.get_mut(k) {
                *s = grown;
            }
            self.editor.resize(&h5, &shape).map_err(IoError::from)?;
        }
        for (name, (h5, at, values)) in ready {
            self.editor
                .write(&h5, &at, &values)
                .map_err(|e| bad(format!("writing {name:?}: {e}")))?;
        }
        Ok(())
    }

    /// A variable's values as the file stores them: its dataset, where they go, the values.
    fn encoded<R: Repr>(
        &mut self,
        name: &Name,
        v: &VariableBase<R>,
        region: &[(Name, usize)],
    ) -> Result<(String, Vec<usize>, annoray_hdf5::Array)> {
        let file_var = self
            .vars
            .get(name)
            .ok_or_else(|| bad(format!("variable {name:?} is not in the file")))?;
        // Values in the file's unit.
        let mut owned = match (&file_var.unit, v.unit()) {
            (Some(u), Some(mine)) if u == mine => v.to_owned(),
            (Some(u), _) => crate::ops::variable_in_unit(v, u)?,
            (None, Some(mine)) if *mine != Unit::one() => {
                return Err(bad(format!(
                    "{name:?}: values in {mine}, but the variable has no unit in the file"
                )));
            }
            (None, _) => v.to_owned(),
        };
        *owned.encoding_mut() = file_var.encoding.clone();
        let raw = crate::cf::encode_variable(name, &owned, Format::Netcdf4)?;
        if raw.dims != file_var.dims {
            return Err(bad(format!(
                "{name:?} is on {:?} in the file, not {:?}",
                file_var.dims, raw.dims
            )));
        }
        let start: Vec<usize> = raw
            .dims
            .iter()
            .map(|d| region.iter().find(|(r, _)| r == d).map_or(0, |(_, s)| *s))
            .collect();
        let mut values = netcdf4::hdf5_array(raw.data.values()?.into_owned())?;
        let h5 = file_var.h5.clone();
        // Characters: padded with NULs to the file's width, so no old characters remain.
        if raw.is_char
            && let annoray_hdf5::Array::U8(chars) = &values
        {
            let width = self
                .editor
                .shape(&h5)
                .map_err(IoError::from)?
                .last()
                .copied()
                .unwrap_or(0);
            let mine = chars.shape().last().copied().unwrap_or(0);
            if mine > width {
                return Err(bad(format!(
                    "{name:?}: strings of {mine} bytes do not fit the file's {width}"
                )));
            }
            if mine < width {
                let mut shape = chars.shape().to_vec();
                if let Some(last) = shape.last_mut() {
                    *last = width;
                }
                let mut padded = ndarray::ArrayD::<u8>::zeros(ndarray::IxDyn(&shape));
                let k = shape.len().saturating_sub(1);
                padded
                    .slice_axis_mut(ndarray::Axis(k), ndarray::Slice::from(0..mine))
                    .assign(chars);
                values = annoray_hdf5::Array::U8(padded);
            }
        }
        Ok((h5, start, values))
    }

    /// Write what is buffered (chunks, indexes, changed shapes).
    ///
    /// # Errors
    /// I/O errors.
    pub fn flush(&mut self) -> Result<()> {
        self.editor.flush().map_err(IoError::from)?;
        Ok(())
    }

    /// The file as it now is, opened lazily ([`flush`](Self::flush)es first).
    ///
    /// # Errors
    /// I/O and decoding errors.
    pub fn dataset(&mut self) -> Result<crate::Dataset> {
        self.flush()?;
        Ok(super::open_dataset_with(&self.path, &OpenOptions::default())?.dataset)
    }

    /// [`flush`](Self::flush) and close.
    ///
    /// # Errors
    /// I/O errors.
    pub fn close(self) -> Result<()> {
        self.editor.close().map_err(IoError::from)?;
        Ok(())
    }
}
