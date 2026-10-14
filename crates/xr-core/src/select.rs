//! Selection helpers shared by `DataArray` and `Dataset`.

use indexmap::IndexMap;

use crate::lookup::resolve;
use crate::{
    CoreError, IndexError, Indexer, IntoIndexer, IntoLabelIndexer, Name, Repr, Result, VariableBase,
};

/// Named indexers, validated: no `..`, no repeated dimension, every dimension in `dims`.
pub(crate) fn named_indexers<I, K, X>(
    indexers: I,
    has_dim: &dyn Fn(&str) -> bool,
) -> Result<Vec<(Name, Indexer)>>
where
    I: IntoIterator<Item = (K, X)>,
    K: AsRef<str>,
    X: IntoIndexer,
{
    let mut out: Vec<(Name, Indexer)> = Vec::new();
    for (dim, ix) in indexers {
        let dim = dim.as_ref();
        if !has_dim(dim) {
            return Err(CoreError::NoSuchDim(dim.into()));
        }
        if out.iter().any(|(d, _)| d == dim) {
            return Err(IndexError::DuplicateDim(dim.into()).into());
        }
        match ix.into_indexer()? {
            Indexer::Ellipsis => return Err(IndexError::EllipsisInNamed.into()),
            ix => out.push((dim.into(), ix)),
        }
    }
    Ok(out)
}

/// Label indexers translated to positional ones using each dimension's coordinate.
pub(crate) fn label_indexers<I, K, X, R>(
    indexers: I,
    has_dim: &dyn Fn(&str) -> bool,
    coords: &IndexMap<Name, VariableBase<R>>,
) -> Result<Vec<(Name, Indexer)>>
where
    I: IntoIterator<Item = (K, X)>,
    K: AsRef<str>,
    X: IntoLabelIndexer,
    R: Repr,
{
    let mut out: Vec<(Name, Indexer)> = Vec::new();
    for (dim, ix) in indexers {
        let dim = dim.as_ref();
        if !has_dim(dim) {
            return Err(CoreError::NoSuchDim(dim.into()));
        }
        if out.iter().any(|(d, _)| d == dim) {
            return Err(IndexError::DuplicateDim(dim.into()).into());
        }
        let coord = coords
            .get(dim)
            .filter(|c| c.dims() == [dim])
            .ok_or_else(|| CoreError::NoIndex(dim.into()))?;
        let name = Name::from(dim);
        let positional = resolve(coord, &name, &ix.into_label_indexer()?)?;
        out.push((name, positional));
    }
    Ok(out)
}

/// Positions of each label along `dim` (fancy label indexing). Partial date strings expand to
/// every position they cover.
pub(crate) fn label_positions<R: Repr>(
    coords: &IndexMap<Name, VariableBase<R>>,
    dim: &str,
    labels: impl IntoIterator<Item = crate::LabelIndexer>,
) -> Result<Vec<isize>> {
    let coord = coords
        .get(dim)
        .filter(|c| c.dims() == [dim])
        .ok_or_else(|| CoreError::NoIndex(dim.into()))?;
    let name = Name::from(dim);
    let len = coord.len();
    let mut out = Vec::new();
    for l in labels {
        match resolve(coord, &name, &l)? {
            Indexer::Int(i) => out.push(i),
            Indexer::Slice(s) => {
                for p in s.indices(len)?.iter() {
                    out.push(isize::try_from(p).map_err(|_| IndexError::Overflow)?);
                }
            }
            Indexer::Ellipsis => return Err(IndexError::EllipsisInNamed.into()),
        }
    }
    Ok(out)
}

/// Apply the indexers whose dimensions `var` has.
pub(crate) fn isel_var<R: Repr>(
    var: VariableBase<R>,
    ix: &[(Name, Indexer)],
) -> Result<VariableBase<R>> {
    let mine: Vec<(Name, Indexer)> = ix
        .iter()
        .filter(|(d, _)| var.dims().contains(d))
        .cloned()
        .collect();
    if mine.is_empty() {
        Ok(var)
    } else {
        var.isel_move(mine)
    }
}

/// Apply indexers to every variable of a map.
pub(crate) fn isel_map<R: Repr>(
    map: IndexMap<Name, VariableBase<R>>,
    ix: &[(Name, Indexer)],
) -> Result<IndexMap<Name, VariableBase<R>>> {
    map.into_iter()
        .map(|(k, v)| Ok((k, isel_var(v, ix)?)))
        .collect()
}

/// Check that `var`'s dimension lengths agree with `sizes`, then record any new ones.
pub(crate) fn merge_sizes<R: Repr>(
    sizes: &mut IndexMap<Name, usize>,
    var: &VariableBase<R>,
) -> Result<()> {
    for (d, n) in var.sizes() {
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
    }
    Ok(())
}
