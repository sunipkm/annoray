//! [`ReadLimits`]: ceilings on what a file may make a reader allocate, for files from untrusted
//! sources.

use std::sync::Arc;

use crate::io::{RawData, RawDataset};
use crate::lazy::LazySource;
use crate::{CoreError, DType, Owned, Result as CoreResult, TypedArray};

/// Ceilings on what opening and reading a file may allocate, checked against the file's
/// metadata before any values are read, and on every read after that. `None` is no limit; the
/// default has none, and [`untrusted`](Self::untrusted) suits files from users or the network.
///
/// Sizes are of values in memory, as declared by the file (its shapes and types), so a small
/// file cannot claim large arrays (or chunks, which a compressed file can expand into) and have
/// them allocated. A breach is [`CoreError::Limit`].
///
/// ```
/// use annoray::io::{OpenOptions, ReadLimits};
///
/// let opts = OpenOptions::default().limits(ReadLimits::untrusted().max_total_bytes(64 << 20));
/// assert_eq!(opts.limits.max_rank, Some(32));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ReadLimits {
    /// The most bytes one read may produce: a variable read whole, a slice of a lazy one, or
    /// one stored chunk (decoded to serve any read that touches it).
    pub max_read_bytes: Option<u64>,
    /// The most bytes of values all variables of the file declare together.
    pub max_total_bytes: Option<u64>,
    /// The most values one variable may have.
    pub max_elements: Option<u64>,
    /// The most dimensions one variable may have.
    pub max_rank: Option<usize>,
    /// The most variables in the file (all groups).
    pub max_variables: Option<usize>,
    /// The most attributes of one variable or group.
    pub max_attributes: Option<usize>,
}

impl ReadLimits {
    /// Limits for files from untrusted sources: reads of at most 256 MiB, 1 GiB of values in
    /// all, 2²⁸ values per variable, 32 dimensions (HDF5's own maximum), 10,000 variables and
    /// 10,000 attributes per variable or group.
    #[must_use]
    pub const fn untrusted() -> Self {
        Self {
            max_read_bytes: Some(256 << 20),
            max_total_bytes: Some(1 << 30),
            max_elements: Some(1 << 28),
            max_rank: Some(32),
            max_variables: Some(10_000),
            max_attributes: Some(10_000),
        }
    }

    /// Sets [`max_read_bytes`](Self::max_read_bytes).
    #[must_use]
    pub const fn max_read_bytes(mut self, n: u64) -> Self {
        self.max_read_bytes = Some(n);
        self
    }

    /// Sets [`max_total_bytes`](Self::max_total_bytes).
    #[must_use]
    pub const fn max_total_bytes(mut self, n: u64) -> Self {
        self.max_total_bytes = Some(n);
        self
    }

    /// Sets [`max_elements`](Self::max_elements).
    #[must_use]
    pub const fn max_elements(mut self, n: u64) -> Self {
        self.max_elements = Some(n);
        self
    }

    /// Sets [`max_rank`](Self::max_rank).
    #[must_use]
    pub const fn max_rank(mut self, n: usize) -> Self {
        self.max_rank = Some(n);
        self
    }

    /// Sets [`max_variables`](Self::max_variables).
    #[must_use]
    pub const fn max_variables(mut self, n: usize) -> Self {
        self.max_variables = Some(n);
        self
    }

    /// Sets [`max_attributes`](Self::max_attributes).
    #[must_use]
    pub const fn max_attributes(mut self, n: usize) -> Self {
        self.max_attributes = Some(n);
        self
    }

    /// Whether no limit is set.
    #[must_use]
    pub fn is_unlimited(&self) -> bool {
        *self == Self::default()
    }

    /// Checks a file's metadata (every group) against the limits, before any values are read.
    ///
    /// # Errors
    /// [`CoreError::Limit`] for the first limit breached.
    pub fn check(&self, raw: &RawDataset<'_>) -> CoreResult<()> {
        let mut totals = Totals::default();
        self.check_group(raw, "", &mut totals)
    }

    /// `raw` with every lazy variable's reads held to [`max_read_bytes`](Self::max_read_bytes).
    #[must_use]
    pub fn wrap(&self, mut raw: RawDataset<'static>) -> RawDataset<'static> {
        if let Some(max) = self.max_read_bytes {
            wrap_group(&mut raw, max);
        }
        raw
    }

    fn check_group(&self, g: &RawDataset<'_>, path: &str, t: &mut Totals) -> CoreResult<()> {
        let at = |name: &str| format!("{path}/{name}");
        exceeds(
            || {
                format!(
                    "attributes of group {:?}",
                    if path.is_empty() { "/" } else { path }
                )
            },
            g.attrs.len() as u64,
            self.max_attributes.map(|n| n as u64),
        )?;
        for (name, v) in &g.vars {
            let shape = v.data.shape();
            exceeds(
                || format!("dimensions of {}", at(name)),
                shape.len() as u64,
                self.max_rank.map(|n| n as u64),
            )?;
            exceeds(
                || format!("attributes of {}", at(name)),
                v.attrs.len() as u64,
                self.max_attributes.map(|n| n as u64),
            )?;
            let size = memory_size(v.data.dtype());
            let elements = product(shape);
            exceeds(
                || format!("values of {}", at(name)),
                elements,
                self.max_elements,
            )?;
            let bytes = elements.saturating_mul(size);
            if let Some(chunks) = &v.storage.chunks {
                exceeds(
                    || format!("bytes of one chunk of {}", at(name)),
                    product(chunks).saturating_mul(size),
                    self.max_read_bytes,
                )?;
            }
            t.variables = t.variables.saturating_add(1);
            t.bytes = t.bytes.saturating_add(bytes);
            exceeds(
                || "variables in the file".to_owned(),
                t.variables as u64,
                self.max_variables.map(|n| n as u64),
            )?;
            exceeds(
                || "bytes of values in the file".to_owned(),
                t.bytes,
                self.max_total_bytes,
            )?;
        }
        for (name, child) in &g.groups {
            self.check_group(child, &at(name), t)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Totals {
    variables: usize,
    bytes: u64,
}

fn exceeds(what: impl FnOnce() -> String, found: u64, limit: Option<u64>) -> CoreResult<()> {
    match limit {
        Some(limit) if found > limit => Err(CoreError::Limit {
            what: what(),
            found,
            limit,
        }),
        _ => Ok(()),
    }
}

/// The number of values of a shape (saturating: a shape too large to count is too large).
fn product(shape: &[usize]) -> u64 {
    shape.iter().fold(1u64, |a, &n| a.saturating_mul(n as u64))
}

/// Bytes per value in memory (text: the string itself, not its characters).
fn memory_size(d: DType) -> u64 {
    use std::mem::size_of;
    let n = match d {
        DType::I8 | DType::U8 | DType::Bool => 1,
        DType::I16 | DType::U16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::Str => size_of::<String>(),
        DType::CfDate => size_of::<crate::CfDate>(),
        DType::C128 => 16,
        _ => 8,
    };
    n as u64
}

fn wrap_group(g: &mut RawDataset<'static>, max: u64) {
    for v in g.vars.values_mut() {
        if let RawData::Lazy(src) = &v.data {
            v.data = RawData::Lazy(Arc::new(Limited {
                inner: src.clone(),
                max,
            }));
        }
    }
    for child in g.groups.values_mut() {
        wrap_group(child, max);
    }
}

/// A lazy source whose reads are held to `max` bytes.
#[derive(Debug)]
struct Limited {
    inner: Arc<dyn LazySource>,
    max: u64,
}

impl LazySource for Limited {
    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> CoreResult<TypedArray<Owned>> {
        exceeds(
            || "bytes of one read".to_owned(),
            product(count).saturating_mul(memory_size(self.dtype())),
            Some(self.max),
        )?;
        self.inner.read_slice(start, count)
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        self.inner.chunking(axis)
    }
}
