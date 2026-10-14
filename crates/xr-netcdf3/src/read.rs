//! Parsing netCDF-3 files. Every read is bounds-checked; malformed files give errors, never
//! panics or allocations larger than the file.

use xr_core::{Attrs, Name};
use xr_io::{IoError, RawDataset, RawDim, RawVariable, Storage};

use crate::ck::{Ck, pad4};
use crate::types::{NC_ATTRIBUTE, NC_DIMENSION, NC_VARIABLE, NcType, decode_array, decode_attr};
use crate::{Version, bad};

const STREAMING: u64 = 0xFFFF_FFFF;

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    v: Version,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("size overflow"))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| bad("unexpected end of header"))?;
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], IoError> {
        self.take(N)?
            .try_into()
            .map_err(|_| bad("unexpected end of header"))
    }

    fn i32(&mut self) -> Result<i32, IoError> {
        Ok(i32::from_be_bytes(self.array()?))
    }

    fn i64(&mut self) -> Result<i64, IoError> {
        Ok(i64::from_be_bytes(self.array()?))
    }

    /// A count or length: 32-bit, or 64-bit in CDF-5.
    fn non_neg(&mut self) -> Result<u64, IoError> {
        let v = if self.v == Version::Cdf5 {
            self.i64()?
        } else {
            i64::from(self.i32()?)
        };
        u64::try_from(v).map_err(|_| bad("negative count"))
    }

    /// A file offset: 32-bit in CDF-1, 64-bit otherwise.
    fn offset(&mut self) -> Result<u64, IoError> {
        let v = if self.v == Version::Cdf1 {
            i64::from(self.i32()?)
        } else {
            self.i64()?
        };
        u64::try_from(v).map_err(|_| bad("negative offset"))
    }

    fn usize(&mut self) -> Result<usize, IoError> {
        usize::try_from(self.non_neg()?).map_err(|_| bad("count too large"))
    }

    fn padded(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let s = self.take(n)?;
        self.take(pad4(n))?;
        Ok(s)
    }

    fn name(&mut self) -> Result<Name, IoError> {
        let n = self.usize()?;
        let bytes = self.padded(n)?;
        let s = std::str::from_utf8(bytes).map_err(|_| bad("name is not UTF-8"))?;
        Ok(Name::from(s))
    }

    /// A list header: `tag count`, or `ABSENT` (zero, zero). Returns the count.
    fn list(&mut self, tag: i32) -> Result<usize, IoError> {
        let t = self.i32()?;
        let n = self.usize()?;
        if t == tag || (t == 0 && n == 0) {
            Ok(n)
        } else {
            Err(bad(format!("expected list tag {tag}, found {t}")))
        }
    }

    fn attrs(&mut self) -> Result<Attrs, IoError> {
        let n = self.list(NC_ATTRIBUTE)?;
        let mut out = Attrs::new();
        for _ in 0..n {
            let name = self.name()?;
            let ty = NcType::from_code(self.i32()?, self.v)
                .ok_or_else(|| bad("unknown attribute type"))?;
            let count = self.usize()?;
            let len = (Ck::from(count) * ty.size())
                .to::<usize>()
                .ok_or_else(|| bad("attribute too large"))?;
            out.insert(name, decode_attr(ty, self.padded(len)?));
        }
        Ok(out)
    }
}

struct VarHeader {
    name: Name,
    dimids: Vec<usize>,
    attrs: Attrs,
    ty: NcType,
    begin: u64,
}

/// Parse a whole netCDF-3 file.
pub(crate) fn parse(buf: &[u8]) -> Result<RawDataset, IoError> {
    let magic = buf.get(..4).ok_or_else(|| bad("file too short"))?;
    let v = match magic {
        b"CDF\x01" => Version::Cdf1,
        b"CDF\x02" => Version::Cdf2,
        b"CDF\x05" => Version::Cdf5,
        _ => return Err(bad("not a netCDF-3 file")),
    };
    let mut c = Cursor { buf, pos: 4, v };
    let numrecs_field = if v == Version::Cdf5 {
        u64::try_from(c.i64()?).unwrap_or(STREAMING)
    } else {
        u64::from(u32::from_be_bytes(c.array()?))
    };

    let ndims = c.list(NC_DIMENSION)?;
    let mut dims: Vec<(Name, u64)> = Vec::new();
    for _ in 0..ndims {
        let name = c.name()?;
        let len = c.non_neg()?;
        dims.push((name, len));
    }
    let record_dim = dims.iter().position(|(_, len)| *len == 0);
    if dims.iter().filter(|(_, len)| *len == 0).count() > 1 {
        return Err(bad("more than one record dimension"));
    }
    let gattrs = c.attrs()?;

    let nvars = c.list(NC_VARIABLE)?;
    let mut vars = Vec::new();
    for _ in 0..nvars {
        let name = c.name()?;
        let nd = c.usize()?;
        let mut dimids = Vec::new();
        for _ in 0..nd {
            let id = c.usize()?;
            if id >= dims.len() {
                return Err(bad(format!(
                    "variable {name:?} uses undefined dimension {id}"
                )));
            }
            dimids.push(id);
        }
        let attrs = c.attrs()?;
        let ty = NcType::from_code(c.i32()?, v)
            .ok_or_else(|| bad(format!("variable {name:?} has an unknown type")))?;
        let _vsize = c.non_neg()?; // recomputed below; may be a sentinel for huge variables
        let begin = c.offset()?;
        vars.push(VarHeader {
            name,
            dimids,
            attrs,
            ty,
            begin,
        });
    }

    // Per-variable slab sizes (one record's worth for record variables).
    let is_record =
        |vh: &VarHeader| record_dim.is_some() && vh.dimids.first().copied() == record_dim;
    let slab = |vh: &VarHeader| -> Result<usize, IoError> {
        let mut n = Ck::from(vh.ty.size());
        for (i, id) in vh.dimids.iter().enumerate() {
            if i == 0 && is_record(vh) {
                continue;
            }
            if Some(*id) == record_dim {
                return Err(bad(format!(
                    "variable {:?} uses the record dimension after the first axis",
                    vh.name
                )));
            }
            let len = dims.get(*id).map_or(0, |(_, l)| *l);
            n = n * len;
        }
        n.to().ok_or_else(|| bad("variable too large"))
    };
    let n_record_vars = vars.iter().filter(|vh| is_record(vh)).count();
    let mut recsize = Ck::from(0usize);
    for vh in vars.iter().filter(|vh| is_record(vh)) {
        let s = slab(vh)?;
        recsize = recsize
            + if n_record_vars == 1 {
                s
            } else {
                (Ck::from(s) + pad4(s)).to::<usize>().unwrap_or(usize::MAX)
            };
    }
    let recsize: usize = recsize.to().ok_or_else(|| bad("record too large"))?;

    let numrecs: usize = if numrecs_field == STREAMING {
        // Streaming file: as many whole records as the file holds.
        let begin_rec = vars
            .iter()
            .filter(|vh| is_record(vh))
            .map(|vh| vh.begin)
            .min()
            .unwrap_or(0);
        let avail = Ck::from(buf.len()) - begin_rec;
        if recsize == 0 {
            0
        } else {
            (avail / recsize).to().unwrap_or(0)
        }
    } else {
        usize::try_from(numrecs_field).map_err(|_| bad("record count too large"))?
    };

    let mut ds = RawDataset {
        attrs: gattrs,
        ..RawDataset::default()
    };
    for (i, (name, len)) in dims.iter().enumerate() {
        let (len, unlimited) = if Some(i) == record_dim {
            (numrecs, true)
        } else {
            (
                usize::try_from(*len).map_err(|_| bad("dimension too large"))?,
                false,
            )
        };
        ds.dims.insert(name.clone(), RawDim { len, unlimited });
    }

    for vh in &vars {
        let slab_len = slab(vh)?;
        let begin = usize::try_from(vh.begin).map_err(|_| bad("offset too large"))?;
        let shape: Vec<usize> = vh
            .dimids
            .iter()
            .map(|id| ds.dims.get_index(*id).map_or(0, |(_, d)| d.len))
            .collect();
        let bytes: Vec<u8> = if is_record(vh) {
            // Never allocate more than the file can hold.
            let total = (Ck::from(slab_len) * numrecs)
                .to::<usize>()
                .filter(|t| *t <= buf.len());
            let mut out = Vec::with_capacity(
                total.ok_or_else(|| bad(format!("{:?}: record data exceeds the file", vh.name)))?,
            );
            for r in 0..numrecs {
                let start = (Ck::from(begin) + Ck::from(r) * recsize)
                    .to::<usize>()
                    .ok_or_else(|| bad("offset overflow"))?;
                let end = start
                    .checked_add(slab_len)
                    .ok_or_else(|| bad("offset overflow"))?;
                out.extend_from_slice(
                    buf.get(start..end).ok_or_else(|| {
                        bad(format!("{:?}: record data past end of file", vh.name))
                    })?,
                );
            }
            out
        } else {
            let end = begin
                .checked_add(slab_len)
                .ok_or_else(|| bad("offset overflow"))?;
            buf.get(begin..end)
                .ok_or_else(|| bad(format!("{:?}: data past end of file", vh.name)))?
                .to_vec()
        };
        let data = decode_array(vh.ty, &bytes, &shape)?;
        let dim_names = vh
            .dimids
            .iter()
            .filter_map(|id| ds.dims.get_index(*id).map(|(n, _)| n.clone()))
            .collect();
        ds.vars.insert(
            vh.name.clone(),
            RawVariable {
                dims: dim_names,
                data,
                is_char: vh.ty == NcType::Char,
                attrs: vh.attrs.clone(),
                storage: Storage::default(),
            },
        );
    }
    Ok(ds)
}
