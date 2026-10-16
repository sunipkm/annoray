//! Groups: hierarchy, attributes, and consolidated metadata.

use serde_json::{Map, Value, json};

use crate::array::{Array, json_of, pretty};
use crate::store::{Store, join};
use crate::{Error, ZarrFormat};

/// A group in a store.
#[derive(Debug, Clone)]
pub struct Group {
    store: Store,
    path: String,
    format: ZarrFormat,
    attrs: Map<String, Value>,
}

/// A member of a group.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Node {
    /// A subgroup.
    Group(Group),
    /// An array.
    Array(Array),
}

/// What is at `path`: a group, an array, or nothing.
/// A node's format, whether it is a group, and its attributes.
type Kind = (ZarrFormat, bool, Map<String, Value>);

fn kind(store: &Store, path: &str) -> Result<Option<Kind>, Error> {
    if let Some(b) = store.get(&join(path, "zarr.json"))? {
        let meta = json_of(&b, "zarr.json")?;
        let attrs = match meta.get("attributes") {
            Some(Value::Object(m)) => m.clone(),
            _ => Map::new(),
        };
        return Ok(match meta.get("node_type").and_then(Value::as_str) {
            Some("group") => Some((ZarrFormat::V3, true, attrs)),
            Some("array") => Some((ZarrFormat::V3, false, attrs)),
            _ => None,
        });
    }
    let attrs = || -> Result<Map<String, Value>, Error> {
        Ok(match store.get(&join(path, ".zattrs"))? {
            Some(a) => match json_of(&a, ".zattrs")? {
                Value::Object(m) => m,
                _ => Map::new(),
            },
            None => Map::new(),
        })
    };
    if store.get(&join(path, ".zgroup"))?.is_some() {
        return Ok(Some((ZarrFormat::V2, true, attrs()?)));
    }
    if store.get(&join(path, ".zarray"))?.is_some() {
        return Ok(Some((ZarrFormat::V2, false, attrs()?)));
    }
    Ok(None)
}

impl Group {
    /// Open the group at `path` (`""` for the root), v2 or v3.
    ///
    /// # Errors
    /// [`Error::NotFound`] if there is no group; [`Error::Invalid`] for malformed metadata.
    pub fn open(store: &Store, path: &str) -> Result<Self, Error> {
        match kind(store, path)? {
            Some((format, true, attrs)) => Ok(Self {
                store: store.clone(),
                path: path.to_owned(),
                format,
                attrs,
            }),
            _ => Err(Error::NotFound(format!(
                "group {path:?} in {}",
                store.root().display()
            ))),
        }
    }

    /// Create the group at `path` with `attrs` (replacing its metadata, keeping members).
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn create(
        store: &Store,
        path: &str,
        format: ZarrFormat,
        attrs: Map<String, Value>,
    ) -> Result<Self, Error> {
        let g = Self {
            store: store.clone(),
            path: path.to_owned(),
            format,
            attrs,
        };
        g.write_meta()?;
        Ok(g)
    }

    fn write_meta(&self) -> Result<(), Error> {
        match self.format {
            ZarrFormat::V2 => {
                self.store.set(
                    &join(&self.path, ".zgroup"),
                    &pretty(&json!({"zarr_format": 2}))?,
                )?;
                self.store.set(
                    &join(&self.path, ".zattrs"),
                    &pretty(&Value::Object(self.attrs.clone()))?,
                )
            }
            ZarrFormat::V3 => self.store.set(
                &join(&self.path, "zarr.json"),
                &pretty(
                    &json!({"zarr_format": 3, "node_type": "group", "attributes": self.attrs}),
                )?,
            ),
        }
    }

    /// The group's path (`""` at the root).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The format.
    #[must_use]
    pub fn format(&self) -> ZarrFormat {
        self.format
    }

    /// The store.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// User attributes.
    #[must_use]
    pub fn attrs(&self) -> &Map<String, Value> {
        &self.attrs
    }

    /// Replace the user attributes.
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn set_attrs(&mut self, attrs: Map<String, Value>) -> Result<(), Error> {
        self.attrs = attrs;
        self.write_meta()
    }

    /// The members, by name (directories that are neither groups nor arrays are skipped).
    ///
    /// # Errors
    /// [`Error::Io`]; [`Error::Invalid`] or [`Error::Unsupported`] for an unreadable array.
    pub fn members(&self) -> Result<Vec<(String, Node)>, Error> {
        let mut out = Vec::new();
        for name in self.store.children(&self.path)? {
            let p = join(&self.path, &name);
            match kind(&self.store, &p)? {
                Some((_, true, _)) => out.push((name, Node::Group(Self::open(&self.store, &p)?))),
                Some((_, false, _)) => out.push((name, Node::Array(Array::open(&self.store, &p)?))),
                None => {}
            }
        }
        Ok(out)
    }

    /// The array `name` in this group.
    ///
    /// # Errors
    /// As for [`Array::open`].
    pub fn array(&self, name: &str) -> Result<Array, Error> {
        Array::open(&self.store, &join(&self.path, name))
    }

    /// The subgroup `name`.
    ///
    /// # Errors
    /// As for [`Group::open`].
    pub fn group(&self, name: &str) -> Result<Self, Error> {
        Self::open(&self.store, &join(&self.path, name))
    }

    /// Write consolidated metadata for everything below this group, so readers that expect
    /// it (xarray, zarr-python) open the store with one read: v2 `.zmetadata`, or v3's
    /// `consolidated_metadata` in this group's `zarr.json` (zarr-python's extension).
    ///
    /// # Errors
    /// [`Error::Io`]; [`Error::Invalid`] for malformed metadata below.
    pub fn consolidate(&self) -> Result<(), Error> {
        let mut entries = Map::new();
        self.collect(&self.path, &mut entries)?;
        match self.format {
            ZarrFormat::V2 => {
                let mut meta = Map::new();
                if let Some(b) = self.store.get(&join(&self.path, ".zgroup"))? {
                    meta.insert(".zgroup".into(), json_of(&b, ".zgroup")?);
                }
                meta.insert(".zattrs".into(), Value::Object(self.attrs.clone()));
                meta.extend(entries);
                let doc = json!({"zarr_consolidated_format": 1, "metadata": meta});
                self.store
                    .set(&join(&self.path, ".zmetadata"), &pretty(&doc)?)
            }
            ZarrFormat::V3 => {
                let doc = json!({
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": self.attrs,
                    "consolidated_metadata": {"kind": "inline", "must_understand": false, "metadata": entries},
                });
                self.store
                    .set(&join(&self.path, "zarr.json"), &pretty(&doc)?)
            }
        }
    }

    fn collect(&self, base: &str, out: &mut Map<String, Value>) -> Result<(), Error> {
        let rel = |p: &str| -> String {
            p.strip_prefix(base)
                .unwrap_or(p)
                .trim_start_matches('/')
                .to_owned()
        };
        for name in self.store.children(base)? {
            let p = join(base, &name);
            match self.format {
                ZarrFormat::V2 => {
                    for file in [".zgroup", ".zarray", ".zattrs"] {
                        if let Some(b) = self.store.get(&join(&p, file))? {
                            out.insert(format!("{}/{file}", rel(&p)), json_of(&b, file)?);
                        }
                    }
                }
                ZarrFormat::V3 => {
                    if let Some(b) = self.store.get(&join(&p, "zarr.json"))? {
                        let mut v = json_of(&b, "zarr.json")?;
                        if let Value::Object(m) = &mut v {
                            m.remove("consolidated_metadata");
                        }
                        out.insert(rel(&p), v);
                    }
                }
            }
            if matches!(kind(&self.store, &p)?, Some((_, true, _))) {
                let base_rel = p.clone();
                let mut sub = Map::new();
                self.collect(&base_rel, &mut sub)?;
                for (k, v) in sub {
                    out.insert(format!("{}/{k}", rel(&p)), v);
                }
            }
        }
        Ok(())
    }
}
