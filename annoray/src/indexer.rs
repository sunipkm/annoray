//! Positional indexers with Python semantics.

use std::fmt;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::str::FromStr;

use crate::{IndexError, Name};

/// A Python-style slice: `start:stop:step`, any part optional, negatives counting from the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PySlice {
    /// First index (inclusive). `None` means the start (or the end, for a negative step).
    pub start: Option<isize>,
    /// Last index (exclusive). `None` means the end (or the start, for a negative step).
    pub stop: Option<isize>,
    /// Stride. `None` means `1`. Must not be `0`.
    pub step: Option<isize>,
}

/// A slice resolved against a concrete axis length: `len` elements, starting at `start`, `step` apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceIndices {
    /// Index of the first selected element. Meaningless when `len == 0`.
    pub start: usize,
    /// Distance between consecutive selected elements. Never `0`.
    pub step: isize,
    /// Number of selected elements.
    pub len: usize,
}

/// One positional indexer: what to select along a single axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Indexer {
    /// Select one element and drop the axis. Negative values count from the end.
    Int(isize),
    /// Select a slice and keep the axis.
    Slice(PySlice),
    /// In positional indexing, stands for as many full slices as needed (Python's `...`).
    Ellipsis,
}

impl PySlice {
    /// The full slice, `:`.
    pub const FULL: Self = Self {
        start: None,
        stop: None,
        step: None,
    };

    /// Construct a slice from its parts.
    #[must_use]
    pub const fn new(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Self {
        Self { start, stop, step }
    }

    /// Resolve against an axis of length `len`, exactly like Python's `slice.indices(len)`.
    /// Out-of-range bounds are clamped.
    ///
    /// # Errors
    /// [`IndexError::ZeroStep`] if the step is zero, [`IndexError::Overflow`] if `len`
    /// exceeds `isize::MAX`.
    pub fn indices(&self, len: usize) -> Result<SliceIndices, IndexError> {
        let step = self.step.unwrap_or(1);
        if step == 0 {
            return Err(IndexError::ZeroStep);
        }
        let n = isize::try_from(len).map_err(|_| IndexError::Overflow)?;
        let (lower, upper) = if step < 0 {
            (-1, n.checked_sub(1).ok_or(IndexError::Overflow)?)
        } else {
            (0, n)
        };
        let clamp = |bound: Option<isize>, default: isize| -> Result<isize, IndexError> {
            Ok(match bound {
                None => default,
                Some(b) if b < 0 => b.checked_add(n).ok_or(IndexError::Overflow)?.max(lower),
                Some(b) => b.min(upper),
            })
        };
        let (start, stop) = if step < 0 {
            (clamp(self.start, upper)?, clamp(self.stop, lower)?)
        } else {
            (clamp(self.start, lower)?, clamp(self.stop, upper)?)
        };

        // Number of elements: ceil(|stop - start| / |step|) when moving in the step's direction.
        let span = if step > 0 {
            stop.checked_sub(start)
        } else {
            start.checked_sub(stop)
        }
        .ok_or(IndexError::Overflow)?;
        let count = match usize::try_from(span) {
            Ok(span) if span > 0 => span
                .checked_sub(1)
                .and_then(|s| s.checked_div(step.unsigned_abs()))
                .and_then(|q| q.checked_add(1))
                .ok_or(IndexError::Overflow)?,
            _ => 0,
        };
        // With count > 0, `start` lies in [0, len); with count == 0 it is unused.
        let start = if count == 0 {
            0
        } else {
            usize::try_from(start).map_err(|_| IndexError::Overflow)?
        };
        Ok(SliceIndices {
            start,
            step,
            len: count,
        })
    }
}

impl SliceIndices {
    /// The selected indices, in order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut next = Some(self.start);
        (0..self.len).map_while(move |_| {
            let cur = next?;
            next = cur.checked_add_signed(self.step);
            Some(cur)
        })
    }

    /// The equivalent ndarray slice.
    ///
    /// # Errors
    /// [`IndexError::Overflow`] if the slice end cannot be represented.
    pub(crate) fn to_ndarray(self) -> Result<ndarray::Slice, IndexError> {
        let ovf = || IndexError::Overflow;
        let start = isize::try_from(self.start).map_err(|_| ovf())?;
        if self.len <= 1 {
            // A single element (or none): the step is irrelevant, so avoid passing extreme
            // steps such as isize::MIN to ndarray.
            let end = start
                .checked_add(isize::try_from(self.len).map_err(|_| ovf())?)
                .ok_or_else(ovf)?;
            return Ok(ndarray::Slice::new(start, Some(end), 1));
        }
        let steps = isize::try_from(self.len.checked_sub(1).ok_or_else(ovf)?).map_err(|_| ovf())?;
        let last = steps
            .checked_mul(self.step)
            .and_then(|d| d.checked_add(start))
            .ok_or_else(ovf)?;
        // ndarray applies a negative step by walking backwards from the end of [start, end).
        Ok(if self.step > 0 {
            ndarray::Slice::new(start, Some(last.checked_add(1).ok_or_else(ovf)?), self.step)
        } else {
            ndarray::Slice::new(last, Some(start.checked_add(1).ok_or_else(ovf)?), self.step)
        })
    }
}

/// Resolve a possibly negative scalar index against an axis of length `len`.
///
/// # Errors
/// [`IndexError::OutOfBounds`] if the index is outside `-len..len`.
pub fn normalize_index(index: isize, len: usize, dim: &Name) -> Result<usize, IndexError> {
    let oob = || IndexError::OutOfBounds {
        dim: dim.clone(),
        index,
        len,
    };
    let resolved = if index < 0 {
        let n = isize::try_from(len).map_err(|_| IndexError::Overflow)?;
        index.checked_add(n).ok_or_else(oob)?
    } else {
        index
    };
    match usize::try_from(resolved) {
        Ok(i) if i < len => Ok(i),
        _ => Err(oob()),
    }
}

impl fmt::Display for PySlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let part = |f: &mut fmt::Formatter<'_>, v: Option<isize>| match v {
            Some(v) => write!(f, "{v}"),
            None => Ok(()),
        };
        part(f, self.start)?;
        f.write_str(":")?;
        part(f, self.stop)?;
        if let Some(step) = self.step {
            write!(f, ":{step}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Indexer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(i) => write!(f, "{i}"),
            Self::Slice(s) => write!(f, "{s}"),
            Self::Ellipsis => f.write_str(".."),
        }
    }
}

impl FromStr for PySlice {
    type Err = IndexError;

    /// Parse `"start:stop"` or `"start:stop:step"`, any part optional (`"::-1"`, `"-5:"`).
    fn from_str(s: &str) -> Result<Self, IndexError> {
        let err = |reason| IndexError::Parse {
            input: s.to_owned(),
            reason,
        };
        if !s.contains(':') {
            return Err(err("a slice needs at least one ':'"));
        }
        let mut parts = s.split(':').map(|p| match p.trim() {
            "" => Ok(None),
            p => p
                .parse()
                .map(Some)
                .map_err(|_| err("bound is not an integer")),
        });
        let start = parts.next().transpose()?.flatten();
        let stop = parts.next().transpose()?.flatten();
        let step = parts.next().transpose()?.flatten();
        if parts.next().is_some() {
            return Err(err("a slice has at most three parts"));
        }
        Ok(Self { start, stop, step })
    }
}

impl FromStr for Indexer {
    type Err = IndexError;

    /// Parse an integer (`"-1"`), a slice (`"1:10:2"`), or an ellipsis (`".."` / `"..."`).
    fn from_str(s: &str) -> Result<Self, IndexError> {
        let t = s.trim();
        if t == ".." || t == "..." {
            Ok(Self::Ellipsis)
        } else if t.contains(':') {
            t.parse().map(Self::Slice)
        } else {
            t.parse().map(Self::Int).map_err(|_| IndexError::Parse {
                input: s.to_owned(),
                reason: "expected an integer, a slice or an ellipsis",
            })
        }
    }
}

impl From<PySlice> for Indexer {
    fn from(s: PySlice) -> Self {
        Self::Slice(s)
    }
}

/// Conversion into an [`Indexer`], used by the indexing methods and macros.
///
/// Implemented for all primitive integers, Rust ranges of them, [`PySlice`] and [`Indexer`].
/// Values that do not fit in `isize` are rejected rather than wrapped.
pub trait IntoIndexer {
    /// Perform the conversion.
    ///
    /// # Errors
    /// [`IndexError::Overflow`] if a value does not fit in `isize`.
    fn into_indexer(self) -> Result<Indexer, IndexError>;
}

fn to_isize<T: TryInto<isize>>(v: T) -> Result<isize, IndexError> {
    v.try_into().map_err(|_| IndexError::Overflow)
}

fn incl_end(end: isize) -> Result<Option<isize>, IndexError> {
    // `..=-1` means "through the last element", i.e. no stop at all.
    if end == -1 {
        Ok(None)
    } else {
        end.checked_add(1).map(Some).ok_or(IndexError::Overflow)
    }
}

impl IntoIndexer for Indexer {
    fn into_indexer(self) -> Result<Indexer, IndexError> {
        Ok(self)
    }
}

impl IntoIndexer for PySlice {
    fn into_indexer(self) -> Result<Indexer, IndexError> {
        Ok(Indexer::Slice(self))
    }
}

impl IntoIndexer for RangeFull {
    fn into_indexer(self) -> Result<Indexer, IndexError> {
        Ok(Indexer::Slice(PySlice::FULL))
    }
}

macro_rules! int_indexer {
    ($($ty:ty),*) => {$(
        impl IntoIndexer for $ty {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                to_isize(self).map(Indexer::Int)
            }
        }
        impl IntoIndexer for Range<$ty> {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                Ok(Indexer::Slice(PySlice::new(Some(to_isize(self.start)?), Some(to_isize(self.end)?), None)))
            }
        }
        impl IntoIndexer for RangeFrom<$ty> {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                Ok(Indexer::Slice(PySlice::new(Some(to_isize(self.start)?), None, None)))
            }
        }
        impl IntoIndexer for RangeTo<$ty> {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                Ok(Indexer::Slice(PySlice::new(None, Some(to_isize(self.end)?), None)))
            }
        }
        impl IntoIndexer for RangeInclusive<$ty> {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                let (start, end) = self.into_inner();
                Ok(Indexer::Slice(PySlice::new(Some(to_isize(start)?), incl_end(to_isize(end)?)?, None)))
            }
        }
        impl IntoIndexer for RangeToInclusive<$ty> {
            fn into_indexer(self) -> Result<Indexer, IndexError> {
                Ok(Indexer::Slice(PySlice::new(None, incl_end(to_isize(self.end)?)?, None)))
            }
        }
    )*};
}

int_indexer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_bound(s: &str) -> Option<isize> {
        if s == "N" {
            None
        } else {
            Some(s.parse().unwrap())
        }
    }

    /// Compare against `range(n)[start:stop:step]` computed by CPython (fixtures/gen_pyslice.py).
    #[test]
    fn matches_python_slice_semantics() {
        let table = include_str!("../../fixtures/pyslice.txt");
        let mut checked = 0;
        for line in table.lines() {
            let (lhs, expected) = line.split_once('|').unwrap();
            let f: Vec<&str> = lhs.split(' ').collect();
            let len: usize = f[0].parse().unwrap();
            let s = PySlice::new(parse_bound(f[1]), parse_bound(f[2]), parse_bound(f[3]));
            let expected: Vec<usize> = expected
                .split(',')
                .filter(|x| !x.is_empty())
                .map(|x| x.parse().unwrap())
                .collect();

            let ix = s.indices(len).unwrap();
            let got: Vec<usize> = ix.iter().collect();
            assert_eq!(got, expected, "len={len} slice={s}");

            // The ndarray translation must select the same elements.
            let a = ndarray::Array1::from_iter(0..len);
            let sliced = a.slice_axis(ndarray::Axis(0), ix.to_ndarray().unwrap());
            assert_eq!(sliced.to_vec(), expected, "ndarray: len={len} slice={s}");
            checked += 1;
        }
        assert_eq!(checked, 6860);
    }

    #[test]
    fn zero_step_and_extremes() {
        assert_eq!(
            PySlice::new(None, None, Some(0)).indices(3),
            Err(IndexError::ZeroStep)
        );
        let all = PySlice::new(Some(isize::MIN), Some(isize::MAX), None)
            .indices(4)
            .unwrap();
        assert_eq!(all.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        let one = PySlice::new(None, None, Some(isize::MIN))
            .indices(4)
            .unwrap();
        assert_eq!(one.iter().collect::<Vec<_>>(), vec![3]);
        assert!(one.to_ndarray().is_ok());
        let rev = PySlice::new(Some(isize::MAX), Some(isize::MIN), Some(-1))
            .indices(3)
            .unwrap();
        assert_eq!(rev.iter().collect::<Vec<_>>(), vec![2, 1, 0]);
    }

    #[test]
    fn scalar_index() {
        let d = Name::from("x");
        assert_eq!(normalize_index(-1, 3, &d), Ok(2));
        assert_eq!(normalize_index(2, 3, &d), Ok(2));
        assert!(matches!(
            normalize_index(3, 3, &d),
            Err(IndexError::OutOfBounds { .. })
        ));
        assert!(matches!(
            normalize_index(-4, 3, &d),
            Err(IndexError::OutOfBounds { .. })
        ));
        assert!(normalize_index(isize::MIN, 3, &d).is_err());
        assert!(normalize_index(0, 0, &d).is_err());
    }

    #[test]
    fn parse() {
        assert_eq!(
            "1:10:2".parse(),
            Ok(Indexer::Slice(PySlice::new(Some(1), Some(10), Some(2))))
        );
        assert_eq!(
            "::-1".parse(),
            Ok(Indexer::Slice(PySlice::new(None, None, Some(-1))))
        );
        assert_eq!(
            "-5:".parse(),
            Ok(Indexer::Slice(PySlice::new(Some(-5), None, None)))
        );
        assert_eq!(":".parse(), Ok(Indexer::Slice(PySlice::FULL)));
        assert_eq!("-3".parse(), Ok(Indexer::Int(-3)));
        assert_eq!("...".parse(), Ok(Indexer::Ellipsis));
        assert!("1:2:3:4".parse::<Indexer>().is_err());
        assert!("a:b".parse::<Indexer>().is_err());
        assert!("".parse::<Indexer>().is_err());
        assert_eq!(PySlice::new(Some(1), None, Some(-2)).to_string(), "1::-2");
    }

    #[test]
    fn ranges() {
        assert_eq!(
            (1..4).into_indexer(),
            Ok(Indexer::Slice(PySlice::new(Some(1), Some(4), None)))
        );
        assert_eq!((..=-1).into_indexer(), Ok(Indexer::Slice(PySlice::FULL)));
        assert_eq!(
            (2..=3).into_indexer(),
            Ok(Indexer::Slice(PySlice::new(Some(2), Some(4), None)))
        );
        assert_eq!(u64::MAX.into_indexer(), Err(IndexError::Overflow));
    }
}
