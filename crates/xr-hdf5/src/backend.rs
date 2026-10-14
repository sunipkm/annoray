//! Plain HDF5 files as raw netCDF-model datasets (dimensions are "phony", as netCDF-C names
//! them, since plain HDF5 has none). netCDF-4 semantics come in `xr-netcdf4`.

use std::path::Path;

use xr_core::{AttrValue, Name, Owned, TypedArray};
use xr_io::{Backend, RawDataset, RawDim, RawVariable, Result, Storage, WriteOptions};

use crate::{AttrData, Attribute, File, Group, Node, WDataset, WGroup, WNode};

/// The pure-Rust HDF5 backend (reading only, for now).
#[derive(Debug, Clone, Copy, Default)]
pub struct Hdf5;

/// An attribute value in the netCDF attribute model (`None` if it cannot be represented).
#[must_use]
pub fn attr_value(a: &Attribute) -> Option<AttrValue> {
    let v = a.value().ok()?;
    let scalar = a.shape.as_ref().is_some_and(Vec::is_empty);
    Some(match v {
        TypedArray::Str(s) if scalar => {
            AttrValue::Str(s.iter().next().cloned().unwrap_or_default())
        }
        TypedArray::Str(s) => AttrValue::Strs(s.iter().cloned().collect()),
        TypedArray::I8(x) => AttrValue::I8(x.iter().copied().collect()),
        TypedArray::I16(x) => AttrValue::I16(x.iter().copied().collect()),
        TypedArray::I32(x) => AttrValue::I32(x.iter().copied().collect()),
        TypedArray::I64(x) => AttrValue::I64(x.iter().copied().collect()),
        TypedArray::U8(x) => AttrValue::U8(x.iter().copied().collect()),
        TypedArray::U16(x) => AttrValue::U16(x.iter().copied().collect()),
        TypedArray::U32(x) => AttrValue::U32(x.iter().copied().collect()),
        TypedArray::U64(x) => AttrValue::U64(x.iter().copied().collect()),
        TypedArray::F32(x) => AttrValue::F32(x.iter().copied().collect()),
        TypedArray::F64(x) => AttrValue::F64(x.iter().copied().collect()),
        _ => return None,
    })
}

fn group_to_raw(g: &Group) -> Result<RawDataset> {
    let mut ds = RawDataset::default();
    for a in g.attrs()? {
        match attr_value(&a) {
            Some(v) => {
                ds.attrs.insert(Name::from(a.name.as_str()), v);
            }
            None => ds.skipped.push((
                Name::from(format!("@{}", a.name)),
                "unsupported attribute type".into(),
            )),
        }
    }
    for (name, node) in g.members()? {
        match node {
            Node::Group(child) => {
                ds.groups
                    .insert(Name::from(name.as_str()), group_to_raw(&child)?);
            }
            Node::Dataset(d) => {
                let data: TypedArray<Owned> = match d.read() {
                    Ok(v) => v,
                    Err(e) => {
                        ds.skipped.push((Name::from(name.as_str()), e.to_string()));
                        continue;
                    }
                };
                // Phony dimensions: reuse one of the same length not already used by this variable.
                let mut dims: Vec<Name> = Vec::new();
                for len in d.shape() {
                    let existing = ds
                        .dims
                        .iter()
                        .find(|(n, dim)| dim.len == len && !dims.contains(n))
                        .map(|(n, _)| n.clone());
                    let dim = existing.unwrap_or_else(|| {
                        let n = Name::from(format!("phony_dim_{}", ds.dims.len()));
                        ds.dims.insert(
                            n.clone(),
                            RawDim {
                                len,
                                unlimited: false,
                            },
                        );
                        n
                    });
                    dims.push(dim);
                }
                let mut attrs = xr_core::Attrs::new();
                for a in d.attrs()? {
                    if let Some(v) = attr_value(&a) {
                        attrs.insert(Name::from(a.name.as_str()), v);
                    }
                }
                let mut storage = Storage::default();
                storage.chunks = d.chunks();
                (storage.deflate, storage.shuffle) = d.compression();
                ds.vars.insert(
                    Name::from(name.as_str()),
                    RawVariable {
                        dims,
                        data,
                        is_char: false,
                        attrs,
                        storage,
                    },
                );
            }
            Node::Other => {}
        }
    }
    Ok(ds)
}

/// A raw dataset as a plain HDF5 tree (dimension names are not stored).
pub(crate) fn raw_to_tree(ds: &RawDataset) -> WGroup {
    let attrs = |a: &xr_core::Attrs| {
        a.iter()
            .map(|(k, v)| (k.to_string(), AttrData::Value(v.clone())))
            .collect()
    };
    let mut members: Vec<(String, WNode)> = ds
        .vars
        .iter()
        .map(|(name, v)| {
            let unlimited = v
                .dims
                .iter()
                .map(|d| ds.dims.get(d).is_some_and(|x| x.unlimited))
                .collect();
            let mut w = WDataset::new(v.data.clone());
            w.char_data = v.is_char;
            w.unlimited = unlimited;
            w.chunks.clone_from(&v.storage.chunks);
            w.deflate = v.storage.deflate;
            w.shuffle = v.storage.shuffle;
            w.attrs = attrs(&v.attrs);
            (name.to_string(), WNode::Dataset(w))
        })
        .collect();
    members.extend(
        ds.groups
            .iter()
            .map(|(n, g)| (n.to_string(), WNode::Group(raw_to_tree(g)))),
    );
    WGroup {
        attrs: attrs(&ds.attrs),
        members,
    }
}

impl Backend for Hdf5 {
    fn name(&self) -> &'static str {
        crate::NAME
    }

    fn read(&self, path: &Path) -> Result<RawDataset> {
        group_to_raw(&File::open(path)?.root()?)
    }

    fn write(&self, ds: &RawDataset, path: &Path, _opts: &WriteOptions) -> Result<()> {
        let bytes = crate::write_file(&raw_to_tree(ds))?;
        std::fs::write(path, bytes).map_err(|e| crate::bad(format!("{}: {e}", path.display())))
    }
}
