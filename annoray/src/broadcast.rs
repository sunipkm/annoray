//! Broadcasting data arrays against each other (xarray's `broadcast` and `broadcast_like`).
//!
//! The arrays are aligned (outer join), then each gains the dimensions it lacks, with their
//! lengths and dimension coordinates from the others. As in xarray, every result has the
//! dimensions in their order of first appearance across the arrays, followed by the array's
//! own excluded dimensions. Data is repeated along the new dimensions (a copy; xarray
//! returns read-only broadcast views).

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
