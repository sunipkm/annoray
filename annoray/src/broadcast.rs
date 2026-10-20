//! Broadcasting data arrays against each other (xarray's `broadcast` and `broadcast_like`).
//!
//! The arrays are aligned (outer join), then each gains the dimensions it lacks, with their
//! lengths and dimension coordinates from the others. As in xarray, every result has the
//! dimensions in their order of first appearance across the arrays, followed by the array's
//! own excluded dimensions. In-memory data is repeated along the new dimensions (a copy;
//! xarray returns read-only broadcast views); lazy data stays lazy, each box read from the
//! source and spread when it is read.

use indexmap::IndexMap;

use crate::{CoreError, DataArray, DataArrayBase, Join, Name, Repr, Result, Variable};

/// `arrays` aligned and broadcast against one another; dimensions in `exclude` are neither
/// added nor aligned-by-broadcast (they are still aligned on labels).
///
/// # Errors
/// Alignment errors; [`CoreError::DimLengthMismatch`] if a dimension has different lengths
/// in different arrays (no labels to align on).
pub fn broadcast<S: Repr>(
    arrays: &[&DataArrayBase<S>],
    exclude: &[&str],
) -> Result<Vec<DataArray>> {
    let aligned = crate::align(arrays, Join::Outer)?;
    let mut sizes: IndexMap<Name, usize> = IndexMap::new();
    let mut coords: IndexMap<Name, Variable> = IndexMap::new();
    for a in &aligned {
        for (d, n) in a.sizes() {
            if exclude.contains(&d.as_str()) {
                continue;
            }
            match sizes.get(d) {
                Some(&m) if m != n => {
                    return Err(CoreError::DimLengthMismatch {
                        dim: d.clone(),
                        expected: m,
                        found: n,
                    });
                }
                Some(_) => {}
                None => {
                    sizes.insert(d.clone(), n);
                }
            }
            if let Some(c) = a.coords().get(d)
                && c.dims() == std::slice::from_ref(d)
            {
                coords.entry(d.clone()).or_insert_with(|| c.clone());
            }
        }
    }
    aligned
        .into_iter()
        .map(|a| {
            if a.variable().is_lazy() {
                return spread_lazily(a, &sizes, &coords, exclude);
            }
            let missing: Vec<(&Name, usize)> = sizes
                .iter()
                .filter(|(d, _)| !a.dims().contains(d))
                .map(|(d, n)| (d, *n))
                .collect();
            let mut out = a;
            for (d, n) in missing.iter().rev() {
                out = out.expand_dims(d, 0)?;
                if *n != 1 {
                    let idx = vec![0_isize; *n];
                    out = out.take(d, &idx)?;
                }
                if let Some(c) = coords.get(*d) {
                    out = out.with_coord((*d).clone(), c.clone())?;
                }
            }
            let order: Vec<&str> = sizes
                .keys()
                .map(Name::as_str)
                .chain(
                    out.dims()
                        .iter()
                        .map(Name::as_str)
                        .filter(|d| exclude.contains(d)),
                )
                .collect();
            out.transpose(&order)
        })
        .collect()
}

impl<S: Repr> DataArrayBase<S> {
    /// This array broadcast against `other` (aligned with it, outer join); see
    /// [`broadcast`].
    ///
    /// # Errors
    /// As for [`broadcast`].
    pub fn broadcast_like<R: Repr>(
        &self,
        other: &DataArrayBase<R>,
        exclude: &[&str],
    ) -> Result<DataArray> {
        let both = broadcast(&[&self.to_owned(), &other.to_owned()], exclude)?;
        both.into_iter()
            .next()
            .ok_or_else(|| CoreError::InvalidReduction("broadcast_like".into()))
    }
}

/// `a` broadcast to `sizes` (then its own excluded dimensions) without reading it.
fn spread_lazily(
    a: DataArray,
    sizes: &IndexMap<Name, usize>,
    coords: &IndexMap<Name, Variable>,
    exclude: &[&str],
) -> Result<DataArray> {
    let own: Vec<(Name, usize)> = a
        .sizes()
        .filter(|(d, _)| exclude.contains(&d.as_str()))
        .map(|(d, n)| (d.clone(), n))
        .collect();
    let (od, shape): (Vec<Name>, Vec<usize>) = sizes
        .iter()
        .map(|(d, n)| (d.clone(), *n))
        .chain(own)
        .unzip();
    let src = SpreadSource {
        src: a.variable().as_lazy_source()?,
        dims: a.dims().to_vec(),
        od: od.clone(),
        shape,
    };
    let mut var = Variable::from_lazy(od.clone(), crate::LazyArray::new(std::sync::Arc::new(src)))?;
    var.set_unit(a.unit().cloned());
    var.take_zone_from(a.variable());
    var.attrs_mut()
        .extend(a.attrs().iter().map(|(k, v)| (k.clone(), v.clone())));
    let (name, _, own_coords) = a.into_parts();
    let mut out = DataArray::new(var);
    for (k, c) in own_coords {
        out = out.with_coord(k, c)?;
    }
    for d in &od {
        if !out.coords().contains_key(d)
            && let Some(c) = coords.get(d)
        {
            out = out.with_coord(d.clone(), c.clone())?;
        }
    }
    Ok(match name {
        Some(n) => out.with_name(n),
        None => out,
    })
}

/// A source spread over dimensions it lacks, read box by box.
#[derive(Debug)]
struct SpreadSource {
    src: std::sync::Arc<dyn crate::LazySource>,
    dims: Vec<Name>,
    od: Vec<Name>,
    shape: Vec<usize>,
}

impl crate::LazySource for SpreadSource {
    fn dtype(&self) -> crate::DType {
        self.src.dtype()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let d = self.od.get(axis)?;
        let a = self.dims.iter().position(|x| x == d)?;
        self.src.chunking(a)
    }
    fn read_slice(
        &self,
        start: &[usize],
        count: &[usize],
    ) -> Result<crate::TypedArray<crate::Owned>> {
        let mut s = Vec::with_capacity(self.dims.len());
        let mut n = Vec::with_capacity(self.dims.len());
        for d in &self.dims {
            let i = self
                .od
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| CoreError::NoSuchDim(d.clone()))?;
            s.push(start.get(i).copied().unwrap_or(0));
            n.push(count.get(i).copied().unwrap_or(0));
        }
        let part = self.src.read_slice(&s, &n)?;
        let sh = ndarray::IxDyn(count);
        let bad = || CoreError::InvalidReduction("broadcast: cannot spread".into());
        Ok(crate::array::typed_map!(&part, x => {
            crate::ops::broadcast_to_dims(x.view(), &self.dims, &self.od)?
                .broadcast(sh.clone())
                .ok_or_else(bad)?
                .to_owned()
        }))
    }
}
