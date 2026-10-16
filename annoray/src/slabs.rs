//! Lane-wise operations on lazy data, slab by slab: operations that need every value of a lane
//! at once (quantiles, `idxmin`, block reductions) read lazy data in slabs along a dimension
//! they keep, each within the streaming budget ([`set_stream_bytes`](crate::set_stream_bytes)),
//! and concatenate the results.

use crate::{
    ConcatOptions, CoreError, DataArray, DataArrayBase, Indexer, PySlice, Repr, Result, concat,
};

/// `f` of `da`, computed slab by slab along `dim` when `da` is lazy. Slabs are a multiple of
/// `multiple` long (block reductions keep whole windows together) and hold about the
/// streaming budget of `f64` values; eager data is passed through in one piece.
pub(crate) fn by_slabs<S: Repr>(
    da: &DataArrayBase<S>,
    dim: &str,
    multiple: usize,
    f: impl Fn(&DataArray) -> Result<DataArray>,
) -> Result<DataArray> {
    let n = da.variable().len_of(dim)?;
    let total: usize = da.variable().shape().iter().product();
    let per_index = total.checked_div(n).unwrap_or(0).saturating_mul(8).max(1);
    let multiple = multiple.max(1);
    let rows = crate::stream_bytes()
        .checked_div(per_index)
        .unwrap_or(1)
        .max(1);
    let rows = rows
        .checked_div(multiple)
        .unwrap_or(1)
        .max(1)
        .saturating_mul(multiple);
    if !da.variable().is_lazy() || rows >= n {
        return f(&da.to_owned());
    }
    let to_i = |x: usize| isize::try_from(x).map_err(|_| CoreError::ArithmeticOverflow("slab"));
    let mut parts = Vec::new();
    let mut start = 0usize;
    while start < n {
        let stop = start.saturating_add(rows).min(n);
        let slab = da
            .isel([(
                dim,
                Indexer::Slice(PySlice::new(Some(to_i(start)?), Some(to_i(stop)?), None)),
            )])?
            .to_owned();
        parts.push(f(&slab)?);
        start = stop;
    }
    let refs: Vec<&DataArray> = parts.iter().collect();
    concat(&refs, dim, &ConcatOptions::default())
}

/// `f` of `da` lane by lane along `dim` (whole lanes at once): lazy data in slabs along
/// another dimension, or whole when `dim` is the only one.
pub(crate) fn by_lanes<S: Repr>(
    da: &DataArrayBase<S>,
    dim: &crate::Name,
    f: impl Fn(&DataArray) -> Result<DataArray>,
) -> Result<DataArray> {
    match da.dims().iter().find(|d| *d != dim) {
        Some(other) => by_slabs(da, other, 1, f),
        None => f(&da.to_owned()),
    }
}
