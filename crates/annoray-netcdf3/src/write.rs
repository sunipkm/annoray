//! Serialising netCDF-3 files.

use std::io::Write;

use annoray_core::{Attrs, Name};
use annoray_io::{Format, IoError, RawData, RawDataset};

use crate::ck::{Ck, pad4};
use crate::types::{
    NC_ATTRIBUTE, NC_DIMENSION, NC_VARIABLE, NcType, encode_array, encode_attr, fill_element,
};
use crate::{Version, bad};

struct Out {
    buf: Vec<u8>,
    v: Version,
}

impl Out {
    fn i32(&mut self, x: i32) {
        self.buf.extend_from_slice(&x.to_be_bytes());
    }

    fn non_neg(&mut self, x: usize) -> Result<(), IoError> {
        if self.v == Version::Cdf5 {
            self.buf.extend_from_slice(
                &i64::try_from(x)
                    .map_err(|_| too_big("count"))?
                    .to_be_bytes(),
            );
        } else {
            self.i32(i32::try_from(x).map_err(|_| too_big("count"))?);
        }
        Ok(())
    }

    fn offset(&mut self, x: usize) -> Result<(), IoError> {
        if self.v == Version::Cdf1 {
            self.i32(
                i32::try_from(x)
                    .map_err(|_| too_big("file offset (use the 64-bit offset format)"))?,
            );
        } else {
            self.buf.extend_from_slice(
                &i64::try_from(x)
                    .map_err(|_| too_big("file offset"))?
                    .to_be_bytes(),
            );
        }
        Ok(())
    }

    fn padded(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.buf.extend(std::iter::repeat_n(0u8, pad4(bytes.len())));
    }

    fn name(&mut self, n: &Name) -> Result<(), IoError> {
        self.non_neg(n.len())?;
        self.padded(n.as_bytes());
        Ok(())
    }

    fn list_header(&mut self, tag: i32, n: usize) -> Result<(), IoError> {
        // An empty list is ABSENT: two zeros.
        self.i32(if n == 0 { 0 } else { tag });
        self.non_neg(n)
    }

    fn attrs(&mut self, attrs: &Attrs, owner: &str) -> Result<(), IoError> {
        self.list_header(NC_ATTRIBUTE, attrs.len())?;
        for (k, a) in attrs {
            let (ty, n, bytes) = encode_attr(a, self.v).ok_or_else(|| {
                bad(format!(
                    "{owner}: attribute {k:?} has a type this format cannot store"
                ))
            })?;
            self.name(k)?;
            self.i32(ty.code());
            self.non_neg(n)?;
            self.padded(&bytes);
        }
        Ok(())
    }
}

fn too_big(what: &str) -> IoError {
    bad(format!("{what} too large for this format"))
}

/// Bytes per read when writing values from a lazy source (whole rows or records at least).
const SLAB_BYTES: usize = 32 << 20;

/// Where a variable's values come from.
enum Values<'a> {
    /// Encoded already (data in memory).
    Bytes(Vec<u8>),
    /// Read, and encoded, a slab at a time.
    Lazy(&'a dyn annoray_core::LazySource),
}

struct Plan<'a> {
    ty: NcType,
    record: bool,
    /// Bytes of one record (record variables) or of the whole variable.
    slab: usize,
    /// `slab` rounded up to four bytes (the header's `vsize`).
    vsize: usize,
    values: Values<'a>,
    /// The variable's shape.
    shape: Vec<usize>,
    /// One element of the fill value, used for padding as netCDF-C does.
    fill: Vec<u8>,
    begin: usize,
}

impl Plan<'_> {
    fn padding(&self, n: usize) -> Vec<u8> {
        self.fill.iter().copied().cycle().take(pad4(n)).collect()
    }

    /// Encoded bytes of rows `[row, row + n)` (along the first dimension; a scalar is one
    /// row).
    fn rows(&self, row: usize, n: usize) -> Result<std::borrow::Cow<'_, [u8]>, IoError> {
        let row_bytes = self.row_bytes()?;
        match &self.values {
            Values::Bytes(b) => {
                let start = (Ck::from(row) * row_bytes)
                    .to::<usize>()
                    .ok_or_else(|| too_big("data"))?;
                let end = (Ck::from(start) + Ck::from(n) * row_bytes)
                    .to::<usize>()
                    .ok_or_else(|| too_big("data"))?;
                Ok(b.get(start..end)
                    .ok_or_else(|| bad("data shorter than its dimensions"))?
                    .into())
            }
            Values::Lazy(src) => {
                let mut start = vec![0usize; self.shape.len()];
                let mut count = self.shape.clone();
                if let (Some(s), Some(c)) = (start.first_mut(), count.first_mut()) {
                    *s = row;
                    *c = n;
                }
                let t = src
                    .read_slice(&start, &count)
                    .map_err(|e| bad(format!("reading data to write: {e}")))?;
                Ok(encode_array(&t)
                    .ok_or_else(|| bad("element type must be CF-encoded first"))?
                    .into())
            }
        }
    }

    /// Bytes of one row (one element for scalars).
    fn row_bytes(&self) -> Result<usize, IoError> {
        self.shape
            .iter()
            .skip(1)
            .try_fold(self.ty.size(), |a, d| a.checked_mul(*d))
            .ok_or_else(|| too_big("variable"))
    }

    /// Rows per read, keeping reads near `SLAB_BYTES`.
    fn rows_per_read(&self) -> Result<usize, IoError> {
        Ok(SLAB_BYTES
            .checked_div(self.row_bytes()?)
            .unwrap_or(1)
            .max(1))
    }
}

fn header(ds: &RawDataset, plans: &[Plan], v: Version, numrecs: usize) -> Result<Vec<u8>, IoError> {
    let mut o = Out { buf: Vec::new(), v };
    o.buf.extend_from_slice(match v {
        Version::Cdf1 => b"CDF\x01",
        Version::Cdf2 => b"CDF\x02",
        Version::Cdf5 => b"CDF\x05",
    });
    o.non_neg(numrecs)?;
    o.list_header(NC_DIMENSION, ds.dims.len())?;
    for (name, d) in &ds.dims {
        o.name(name)?;
        o.non_neg(if d.unlimited { 0 } else { d.len })?;
    }
    o.attrs(&ds.attrs, "global")?;
    o.list_header(NC_VARIABLE, ds.vars.len())?;
    for ((name, var), p) in ds.vars.iter().zip(plans) {
        o.name(name)?;
        o.non_neg(var.dims.len())?;
        for d in &var.dims {
            o.non_neg(
                ds.dims
                    .get_index_of(d)
                    .ok_or_else(|| bad(format!("{name:?}: unknown dimension {d:?}")))?,
            )?;
        }
        o.attrs(&var.attrs, name)?;
        o.i32(p.ty.code());
        // vsize saturates for huge variables; readers recompute it.
        match v {
            Version::Cdf5 => o.non_neg(p.vsize)?,
            _ => o
                .buf
                .extend_from_slice(&u32::try_from(p.vsize).unwrap_or(u32::MAX).to_be_bytes()),
        }
        o.offset(p.begin)?;
    }
    Ok(o.buf)
}

/// Serialise a dataset as a netCDF-3 file in memory.
pub(crate) fn serialize(ds: &RawDataset, format: Format) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::new();
    write_to(ds, format, &mut out)?;
    Ok(out)
}

/// Write a dataset as a netCDF-3 file to `out`, in order. Lazy variables are read and
/// written a slab (or a batch of records) at a time, so they need never be in memory at once.
pub(crate) fn write_to(
    ds: &RawDataset,
    format: Format,
    out: &mut dyn Write,
) -> Result<(), IoError> {
    let v = match format {
        Format::Netcdf3Classic => Version::Cdf1,
        Format::Netcdf3Offset64 => Version::Cdf2,
        Format::Netcdf3Data64 => Version::Cdf5,
        other => {
            return Err(IoError::Format {
                what: "file".into(),
                format: other,
                reason: "not a netCDF-3 format".into(),
            });
        }
    };
    let fmt_err = |what: String, reason: &str| IoError::Format {
        what,
        format,
        reason: reason.into(),
    };
    if !ds.groups.is_empty() {
        return Err(fmt_err("groups".into(), "netCDF-3 has no groups"));
    }
    let unlimited: Vec<&Name> = ds
        .dims
        .iter()
        .filter(|(_, d)| d.unlimited)
        .map(|(n, _)| n)
        .collect();
    if unlimited.len() > 1 {
        return Err(fmt_err(
            "unlimited dimensions".into(),
            "netCDF-3 allows only one",
        ));
    }
    let record_dim = unlimited.first().copied();
    let numrecs = record_dim.and_then(|d| ds.dims.get(d)).map_or(0, |d| d.len);

    let mut plans = Vec::new();
    for (name, var) in &ds.vars {
        let ty = NcType::of(var.data.dtype(), var.is_char, v).ok_or_else(|| {
            fmt_err(
                format!("variable {name:?} ({})", var.data.dtype()),
                "type not available",
            )
        })?;
        let record = record_dim.is_some_and(|r| var.dims.first() == Some(r));
        if var.dims.iter().skip(1).any(|d| Some(d) == record_dim) {
            return Err(fmt_err(
                format!("variable {name:?}"),
                "the unlimited dimension must be the first",
            ));
        }
        let expected: Vec<usize> = var
            .dims
            .iter()
            .map(|d| ds.dims.get(d).map_or(0, |x| x.len))
            .collect();
        if var.data.shape() != expected.as_slice() {
            return Err(bad(format!(
                "{name:?}: data shape {:?} does not match its dimensions {expected:?}",
                var.data.shape()
            )));
        }
        let mut slab = Ck::from(ty.size());
        for (i, n) in expected.iter().enumerate() {
            if !(record && i == 0) {
                slab = slab * *n;
            }
        }
        let slab: usize = slab.to().ok_or_else(|| too_big("variable"))?;
        let vsize = (Ck::from(slab) + pad4(slab))
            .to()
            .ok_or_else(|| too_big("variable"))?;
        let values =
            match &var.data {
                RawData::Lazy(src) => Values::Lazy(src.as_ref()),
                RawData::Loaded(a) => Values::Bytes(encode_array(a).ok_or_else(|| {
                    bad(format!("{name:?}: element type must be CF-encoded first"))
                })?),
            };
        let fill = fill_element(ty, var.attrs.get("_FillValue"));
        plans.push(Plan {
            ty,
            record,
            slab,
            vsize,
            values,
            shape: expected,
            fill,
            begin: 0,
        });
    }

    // Lay out data after the header (whose size doesn't depend on the offsets).
    let header_len = header(ds, &plans, v, numrecs)?.len();
    let n_record = plans.iter().filter(|p| p.record).count();
    let mut cur = Ck::from(header_len);
    for p in plans.iter_mut().filter(|p| !p.record) {
        p.begin = cur.to().ok_or_else(|| too_big("file"))?;
        cur = cur + p.vsize;
    }
    let mut recsize = Ck::from(0usize);
    for p in plans.iter_mut().filter(|p| p.record) {
        p.begin = (cur + recsize).to().ok_or_else(|| too_big("file"))?;
        // A lone record variable is not padded between records.
        recsize = recsize + if n_record == 1 { p.slab } else { p.vsize };
    }
    recsize.to::<usize>().ok_or_else(|| too_big("record"))?;

    let io = |e: std::io::Error| bad(format!("writing: {e}"));
    out.write_all(&header(ds, &plans, v, numrecs)?)
        .map_err(io)?;
    // Fixed-size variables, each followed by its padding.
    for p in plans.iter().filter(|p| !p.record) {
        let rows = p.shape.first().copied().unwrap_or(1);
        let step = p.rows_per_read()?;
        let mut r = 0usize;
        while r < rows {
            let n = step.min(rows.saturating_sub(r));
            out.write_all(&p.rows(r, n)?).map_err(io)?;
            r = r.saturating_add(n);
        }
        out.write_all(&p.padding(p.slab)).map_err(io)?;
    }
    // Records, interleaved: read a batch of records of every record variable at a time.
    let records: Vec<&Plan<'_>> = plans.iter().filter(|p| p.record).collect();
    let recsize = records
        .iter()
        .try_fold(0usize, |a, p| a.checked_add(p.vsize))
        .ok_or_else(|| too_big("record"))?;
    let batch = SLAB_BYTES.checked_div(recsize).unwrap_or(1).max(1);
    let mut r0 = 0usize;
    while r0 < numrecs {
        let n = batch.min(numrecs.saturating_sub(r0));
        let slabs = records
            .iter()
            .map(|p| p.rows(r0, n))
            .collect::<Result<Vec<_>, _>>()?;
        for i in 0..n {
            for (p, bytes) in records.iter().zip(&slabs) {
                let start = (Ck::from(i) * p.slab)
                    .to::<usize>()
                    .ok_or_else(|| too_big("record"))?;
                let end = start.checked_add(p.slab).ok_or_else(|| too_big("record"))?;
                out.write_all(
                    bytes
                        .get(start..end)
                        .ok_or_else(|| bad("record data shorter than its dimensions"))?,
                )
                .map_err(io)?;
                if n_record != 1 {
                    out.write_all(&p.padding(p.slab)).map_err(io)?;
                }
            }
        }
        r0 = r0.saturating_add(n);
    }
    Ok(())
}
