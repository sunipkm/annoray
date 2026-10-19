//! HDF5 datatypes and conversion of raw element bytes to typed arrays.

use std::collections::HashMap;

use crate::Error;
use crate::{Array, ValueType};
use ndarray::{ArrayD, IxDyn};

use crate::cursor::{Cursor, Sizes, align8};
use crate::format::{kind, object_header};
use crate::{File, bad};

/// String padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrPad {
    /// Terminated by a NUL (anything after it is ignored).
    NullTerm,
    /// Padded with NULs.
    NullPad,
    /// Padded with spaces.
    SpacePad,
}

/// An HDF5 datatype, as far as annoray can represent it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// A two's-complement or unsigned integer.
    Int {
        /// Bytes per element (1, 2, 4 or 8).
        size: u8,
        /// Whether signed.
        signed: bool,
        /// Whether stored big-endian.
        big_endian: bool,
    },
    /// An integer whose value occupies only `precision` bits starting at bit `offset` of its
    /// `size` bytes (HDF5 "bit field" precision; what the N-bit filter writes).
    PackedInt {
        /// Bytes per element (1, 2, 4 or 8).
        size: u8,
        /// Whether signed (the field is sign-extended).
        signed: bool,
        /// Whether stored big-endian.
        big_endian: bool,
        /// The first significant bit.
        offset: u16,
        /// The number of significant bits.
        precision: u16,
    },
    /// An IEEE float.
    Float {
        /// Bytes per element (4 or 8).
        size: u8,
        /// Whether stored big-endian.
        big_endian: bool,
    },
    /// A fixed-length string.
    FixedString {
        /// Bytes per element.
        size: usize,
        /// Padding.
        pad: StrPad,
    },
    /// A variable-length string (stored in the global heap).
    VarString {
        /// Padding.
        pad: StrPad,
    },
    /// An enumeration over an integer type; values are read as the base type.
    Enum {
        /// The base integer type.
        base: Box<Datatype>,
        /// `(name, value bytes)` pairs.
        members: Vec<(String, Vec<u8>)>,
    },
    /// An object reference (the address of an object header).
    ObjectReference,
    /// A variable-length sequence of object references (e.g. `DIMENSION_LIST`).
    ReferenceList,
    /// A compound (record) type: named members at byte offsets; read with
    /// [`Dataset::read_compound`](crate::Dataset::read_compound).
    Compound {
        /// Bytes per record.
        size: usize,
        /// The members, in declaration order.
        members: Vec<Member>,
    },
    /// A fixed-size array of another type (as a compound member).
    ArrayOf {
        /// The element type.
        base: Box<Datatype>,
        /// The array's dimensions.
        dims: Vec<usize>,
    },
    /// A variable-length sequence of numbers (a ragged array: netCDF `vlen`, h5py
    /// `vlen_dtype`); read with [`Dataset::read_ragged`](crate::Dataset::read_ragged).
    VarSeq {
        /// The element type (an integer, float or enumeration).
        base: Box<Datatype>,
    },
    /// Anything else (compound, region reference, opaque, array, bitfield, …): not converted.
    Other {
        /// The HDF5 type class.
        class: u8,
        /// Bytes per element.
        size: usize,
    },
}

fn pad_of(bits: u8) -> StrPad {
    match bits & 0x0F {
        0 => StrPad::NullTerm,
        2 => StrPad::SpacePad,
        _ => StrPad::NullPad,
    }
}

impl Datatype {
    /// The element type values are read as, or `None` if they cannot be read.
    #[must_use]
    pub fn value_dtype(&self) -> Option<ValueType> {
        Some(match self {
            Self::Int {
                size: 1,
                signed: true,
                ..
            } => ValueType::I8,
            Self::Int { size: 1, .. } => ValueType::U8,
            Self::Int {
                size: 2,
                signed: true,
                ..
            } => ValueType::I16,
            Self::Int { size: 2, .. } => ValueType::U16,
            Self::Int {
                size: 4,
                signed: true,
                ..
            } => ValueType::I32,
            Self::Int { size: 4, .. } => ValueType::U32,
            Self::Int {
                size: 8,
                signed: true,
                ..
            } => ValueType::I64,
            Self::Int { .. } => ValueType::U64,
            // Half precision is read as f32 (exactly).
            Self::Float { size: 2 | 4, .. } => ValueType::F32,
            Self::Float { .. } => ValueType::F64,
            Self::FixedString { .. } | Self::VarString { .. } => ValueType::Str,
            Self::Enum { base, .. } => return base.value_dtype(),
            Self::PackedInt {
                size,
                signed,
                big_endian,
                ..
            } => {
                return Self::Int {
                    size: *size,
                    signed: *signed,
                    big_endian: *big_endian,
                }
                .value_dtype();
            }
            Self::ObjectReference
            | Self::ReferenceList
            | Self::VarSeq { .. }
            | Self::Compound { .. }
            | Self::ArrayOf { .. }
            | Self::Other { .. } => return None,
        })
    }

    /// Parse a datatype message.
    pub(crate) fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut c = Cursor::new(
            data,
            0,
            Sizes {
                offset: 8,
                length: 8,
            },
        );
        Self::parse_from(&mut c)
    }

    fn parse_from(c: &mut Cursor<'_>) -> Result<Self, Error> {
        let cv = c.u8()?;
        let (class, version) = (cv & 0x0F, cv >> 4);
        let bits = c.bytes(3)?;
        let (b0, b1) = (
            bits.first().copied().unwrap_or(0),
            bits.get(1).copied().unwrap_or(0),
        );
        let size = usize::try_from(c.u32()?).map_err(|_| bad("datatype too large"))?;
        Ok(match class {
            0 => {
                let offset = c.u16()?;
                let precision = c.u16()?;
                let (signed, big_endian) = (b0 & 0x08 != 0, b0 & 1 != 0);
                match u8::try_from(size) {
                    Ok(s @ (1 | 2 | 4 | 8))
                        if offset == 0 && Some(usize::from(precision)) == size.checked_mul(8) =>
                    {
                        Self::Int {
                            size: s,
                            signed,
                            big_endian,
                        }
                    }
                    Ok(s @ (1 | 2 | 4 | 8))
                        if precision > 0
                            && usize::from(offset).checked_add(usize::from(precision))
                                <= size.checked_mul(8) =>
                    {
                        Self::PackedInt {
                            size: s,
                            signed,
                            big_endian,
                            offset,
                            precision,
                        }
                    }
                    _ => Self::Other { class, size },
                }
            }
            1 => {
                c.skip(12)?;
                match u8::try_from(size) {
                    Ok(s @ (2 | 4 | 8)) => Self::Float {
                        size: s,
                        big_endian: b0 & 1 != 0,
                    },
                    _ => Self::Other { class, size },
                }
            }
            6 => {
                let n = usize::from(u16::from_le_bytes([b0, b1]));
                let mut members = Vec::with_capacity(n.min(4096));
                // Version 3 stores offsets in as few bytes as the record size needs.
                let offset_bytes = match size {
                    0..=0xff => 1,
                    0x100..=0xffff => 2,
                    0x1_0000..=0xff_ffff => 3,
                    _ => 4,
                };
                for _ in 0..n {
                    let rest = c.rest();
                    let len = rest
                        .iter()
                        .position(|b| *b == 0)
                        .ok_or_else(|| bad("bad compound member name"))?;
                    let name =
                        String::from_utf8_lossy(rest.get(..len).unwrap_or_default()).into_owned();
                    let stored = if version >= 3 {
                        len.saturating_add(1)
                    } else {
                        align8(len.saturating_add(1)).ok_or_else(|| bad("bad compound"))?
                    };
                    c.skip(stored)?;
                    let offset = if version >= 3 {
                        c.uvar(offset_bytes)?
                    } else {
                        u64::from(c.u32()?)
                    };
                    let offset =
                        usize::try_from(offset).map_err(|_| bad("compound member offset"))?;
                    // Version 1 members carry their own array dimensions (up to four).
                    let mut dims = Vec::new();
                    if version == 1 {
                        let rank = usize::from(c.u8()?);
                        c.skip(3 + 4 + 4)?; // reserved, permutation, reserved
                        for i in 0..4 {
                            let d = usize::try_from(c.u32()?).map_err(|_| bad("member dims"))?;
                            if i < rank {
                                dims.push(d);
                            }
                        }
                    }
                    let mut dtype = Self::parse_from(c)?;
                    if !dims.is_empty() {
                        dtype = Self::ArrayOf {
                            base: Box::new(dtype),
                            dims,
                        };
                    }
                    members.push(Member {
                        name,
                        offset,
                        dtype,
                    });
                }
                Self::Compound { size, members }
            }
            10 => {
                let rank = usize::from(c.u8()?);
                if version < 3 {
                    c.skip(3)?;
                }
                let mut dims = Vec::with_capacity(rank.min(32));
                for _ in 0..rank {
                    dims.push(usize::try_from(c.u32()?).map_err(|_| bad("array dims"))?);
                }
                if version < 3 {
                    c.skip(rank.saturating_mul(4))?; // permutation
                }
                Self::ArrayOf {
                    base: Box::new(Self::parse_from(c)?),
                    dims,
                }
            }
            3 => Self::FixedString {
                size,
                pad: pad_of(b0),
            },
            8 => {
                let base = Self::parse_from(c)?;
                let n = usize::from(u16::from_le_bytes([b0, b1]));
                let mut names = Vec::with_capacity(n.min(65_536));
                for _ in 0..n {
                    let rest = c.rest();
                    let len = rest
                        .iter()
                        .position(|b| *b == 0)
                        .ok_or_else(|| bad("bad enum name"))?;
                    let name =
                        String::from_utf8_lossy(rest.get(..len).unwrap_or_default()).into_owned();
                    let stored = if version >= 3 {
                        len.saturating_add(1)
                    } else {
                        align8(len.saturating_add(1)).ok_or_else(|| bad("bad enum"))?
                    };
                    c.skip(stored)?;
                    names.push(name);
                }
                let vsize = base.size();
                let members = names
                    .into_iter()
                    .map(|nm| Ok((nm, c.bytes(vsize)?.to_vec())))
                    .collect::<Result<Vec<_>, Error>>()?;
                Self::Enum {
                    base: Box::new(base),
                    members,
                }
            }
            9 => {
                let base = Self::parse_from(c)?;
                if b0 & 0x0F == 1 {
                    Self::VarString {
                        pad: pad_of(b0 >> 4),
                    }
                } else if base == Self::ObjectReference {
                    Self::ReferenceList
                } else if base.value_dtype().is_some_and(|v| v != ValueType::Str) {
                    Self::VarSeq {
                        base: Box::new(base),
                    }
                } else {
                    Self::Other { class, size }
                }
            }
            7 if b0 & 0x0F == 0 && size == 8 => Self::ObjectReference,
            2 | 4 | 5 | 7 => {
                // Properties of unsupported classes are not needed: messages are length-delimited.
                Self::Other { class, size }
            }
            other => return Err(bad(format!("unknown datatype class {other}"))),
        })
    }

    /// Bytes per element as stored in a dataset or attribute.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Int { size, .. } | Self::PackedInt { size, .. } | Self::Float { size, .. } => {
                usize::from(*size)
            }
            Self::FixedString { size, .. } | Self::Other { size, .. } => *size,
            // Length (4) + global heap collection address + object index (4); sized at the
            // file's offset width by `element_size`.
            Self::VarString { .. } | Self::ReferenceList | Self::VarSeq { .. } => 16,
            Self::ObjectReference => 8,
            Self::Enum { base, .. } => base.size(),
            Self::Compound { size, .. } => *size,
            Self::ArrayOf { base, dims } => dims
                .iter()
                .try_fold(base.size(), |a, d| a.checked_mul(*d))
                .unwrap_or(usize::MAX),
        }
    }

    /// Bytes per element in a file with offset width `offset`.
    pub(crate) fn element_size(&self, offset: u8) -> usize {
        match self {
            Self::VarString { .. } | Self::ReferenceList | Self::VarSeq { .. } => {
                8usize.saturating_add(usize::from(offset))
            }
            other => other.size(),
        }
    }
}

/// A member of a compound type.
#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    /// Its name.
    pub name: String,
    /// Its byte offset in a record.
    pub offset: usize,
    /// Its type.
    pub dtype: Datatype,
}

/// One field of a (possibly nested) compound type: its dotted path, byte offset in the record,
/// element type and any array dimensions.
pub(crate) struct Field<'a> {
    pub(crate) path: String,
    pub(crate) offset: usize,
    pub(crate) dtype: &'a Datatype,
    pub(crate) dims: Vec<usize>,
}

/// The leaf fields of a compound type, nested compounds flattened with dotted paths.
pub(crate) fn fields<'a>(
    members: &'a [Member],
    prefix: &str,
    base: usize,
    out: &mut Vec<Field<'a>>,
) -> Result<(), Error> {
    for m in members {
        let path = if prefix.is_empty() {
            m.name.clone()
        } else {
            format!("{prefix}.{}", m.name)
        };
        let offset = base
            .checked_add(m.offset)
            .ok_or_else(|| bad("compound member offset"))?;
        match &m.dtype {
            Datatype::Compound { members, .. } => fields(members, &path, offset, out)?,
            Datatype::ArrayOf { base: b, dims } => match b.as_ref() {
                Datatype::Compound { .. } | Datatype::ArrayOf { .. } => {
                    return Err(Error::Unsupported(format!(
                        "compound member {path:?}: arrays of records"
                    )));
                }
                inner => out.push(Field {
                    path,
                    offset,
                    dtype: inner,
                    dims: dims.clone(),
                }),
            },
            other => out.push(Field {
                path,
                offset,
                dtype: other,
                dims: Vec::new(),
            }),
        }
    }
    Ok(())
}

/// The leaf fields of a compound type's members: each one's dotted path, element type and
/// array dimensions (empty for a scalar member), in record order.
///
/// # Errors
/// Arrays of records, which have no flat leaves, and offsets past `usize`.
pub fn compound_fields(members: &[Member]) -> Result<Vec<(String, Datatype, Vec<usize>)>, Error> {
    let mut out = Vec::new();
    fields(members, "", 0, &mut out)?;
    Ok(out
        .into_iter()
        .map(|f| (f.path, f.dtype.clone(), f.dims))
        .collect())
}

/// Resolve a shared (committed) datatype.
pub(crate) fn shared_datatype(file: &File, data: &[u8]) -> Result<Datatype, Error> {
    Ok(shared_datatype_at(file, data)?.0)
}

/// A shared (committed) datatype and the address of its object header.
pub(crate) fn shared_datatype_at(file: &File, data: &[u8]) -> Result<(Datatype, u64), Error> {
    let mut c = Cursor::new(data, 0, file.sizes);
    let version = c.u8()?;
    let kind_ = c.u8()?;
    if version == 1 {
        c.skip(6)?;
    }
    if version == 3 && kind_ == 1 {
        return Err(bad(
            "datatypes in the shared object header message heap are not supported",
        ));
    }
    let addr = c
        .addr()?
        .ok_or_else(|| bad("undefined shared datatype address"))?;
    let msgs = object_header(file, addr)?;
    let m = msgs
        .iter()
        .find(|m| m.kind == kind::DATATYPE)
        .ok_or_else(|| bad("committed datatype without a datatype message"))?;
    Ok((Datatype::parse(&m.data)?, addr))
}

fn strip(bytes: &[u8], pad: StrPad) -> String {
    let end = match pad {
        StrPad::NullTerm => bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len()),
        StrPad::NullPad => bytes
            .iter()
            .rposition(|b| *b != 0)
            .map_or(0, |p| p.saturating_add(1)),
        StrPad::SpacePad => bytes
            .iter()
            .rposition(|b| *b != b' ' && *b != 0)
            .map_or(0, |p| p.saturating_add(1)),
    };
    String::from_utf8_lossy(bytes.get(..end).unwrap_or_default()).into_owned()
}

/// Objects of one global heap collection, by index.
pub(crate) fn global_heap(file: &File, addr: u64) -> Result<HashMap<u16, Vec<u8>>, Error> {
    let mut c = file.cursor(addr)?;
    let start = c.pos();
    c.expect(b"GCOL")?;
    c.skip(4)?; // version, reserved
    let size = c.len_usize()?; // includes this header
    let end = start
        .checked_add(size)
        .ok_or_else(|| bad("bad global heap size"))?;
    let mut out = HashMap::new();
    let header = 8usize.saturating_add(usize::from(file.sizes.length));
    while c.pos().saturating_add(header) <= end {
        let index = c.u16()?;
        if index == 0 {
            break; // free space
        }
        c.skip(6)?; // reference count, reserved
        let n = c.len_usize()?;
        let data = c.bytes(n)?.to_vec();
        c.skip(
            align8(n)
                .ok_or_else(|| bad("bad global heap"))?
                .saturating_sub(n),
        )?;
        out.insert(index, data);
    }
    Ok(out)
}

macro_rules! ints {
    ($bytes:expr, $t:ty, $n:literal, $be:expr) => {{
        let one = |c: &[u8]| {
            let a: [u8; $n] = c.try_into().unwrap_or([0; $n]);
            if $be {
                <$t>::from_be_bytes(a)
            } else {
                <$t>::from_le_bytes(a)
            }
        };
        decode_all!($bytes, $n, one, $t)
    }};
}

/// Every `$n`-byte element of `$bytes` through `$one`: in parallel for large inputs (where
/// the copy, and the page faults of a mapped file, dominate) with the `rayon` feature.
#[cfg(feature = "rayon")]
macro_rules! decode_all {
    ($bytes:expr, $n:literal, $one:expr, $t:ty) => {{
        use rayon::prelude::*;
        let b: &[u8] = $bytes;
        if b.len() >= PARALLEL_BYTES {
            // Blocks of plain loops (which vectorise), each thread filling its own part.
            const BLOCK: usize = 1 << 18;
            let mut out = vec![<$t>::default(); b.len() / $n];
            out.par_chunks_mut(BLOCK)
                .zip(b.par_chunks(BLOCK * $n))
                .for_each(|(o, b)| {
                    for (o, c) in o.iter_mut().zip(b.chunks_exact($n)) {
                        *o = $one(c);
                    }
                });
            out
        } else {
            b.chunks_exact($n).map($one).collect::<Vec<$t>>()
        }
    }};
}

#[cfg(not(feature = "rayon"))]
macro_rules! decode_all {
    ($bytes:expr, $n:literal, $one:expr, $t:ty) => {{
        let b: &[u8] = $bytes;
        b.chunks_exact($n).map($one).collect::<Vec<$t>>()
    }};
}

/// Inputs at least this large are decoded in parallel.
#[cfg(feature = "rayon")]
const PARALLEL_BYTES: usize = 1 << 22;

/// The `precision`-bit field at bit `offset` of every integer, sign-extended if `signed` (what
/// HDF5's conversion of a bit-field integer type to a native one gives).
fn unpack_bits(data: Array, offset: u16, precision: u16, signed: bool) -> Array {
    let (offset, precision) = (u32::from(offset), u32::from(precision));
    let field = move |raw: u64| -> i128 {
        let v = raw.checked_shr(offset).unwrap_or(0)
            & (1u64.checked_shl(precision).unwrap_or(0).wrapping_sub(1));
        let top = 1u64.checked_shl(precision.saturating_sub(1)).unwrap_or(0);
        if signed && precision > 0 && v & top != 0 {
            // Two's complement in `precision` bits; both values are below 2^64.
            i128::from(v).saturating_sub(i128::from(top).saturating_mul(2))
        } else {
            i128::from(v)
        }
    };
    macro_rules! go {
        ($a:expr, $t:ty, $u:ty) => {
            $a.mapv(|x: $t| {
                let raw = u64::from(<$u>::from_ne_bytes(x.to_ne_bytes()));
                <$t>::try_from(field(raw)).unwrap_or(0)
            })
        };
    }
    match data {
        Array::I8(a) => Array::I8(go!(a, i8, u8)),
        Array::U8(a) => Array::U8(go!(a, u8, u8)),
        Array::I16(a) => Array::I16(go!(a, i16, u16)),
        Array::U16(a) => Array::U16(go!(a, u16, u16)),
        Array::I32(a) => Array::I32(go!(a, i32, u32)),
        Array::U32(a) => Array::U32(go!(a, u32, u32)),
        Array::I64(a) => Array::I64(go!(a, i64, u64)),
        Array::U64(a) => Array::U64(go!(a, u64, u64)),
        other => other,
    }
}

/// Convert raw element bytes into an array of `shape`.
pub(crate) fn decode(
    file: &File,
    dt: &Datatype,
    bytes: &[u8],
    shape: &[usize],
) -> Result<Array, Error> {
    let sh = IxDyn(shape);
    let err = |e: ndarray::ShapeError| bad(format!("data does not match its shape: {e}"));
    Ok(match dt {
        Datatype::Int {
            size,
            signed,
            big_endian: be,
        } => match (size, signed) {
            (1, true) => {
                Array::I8(ArrayD::from_shape_vec(sh, ints!(bytes, i8, 1, *be)).map_err(err)?)
            }
            (1, false) => Array::U8(ArrayD::from_shape_vec(sh, bytes.to_vec()).map_err(err)?),
            (2, true) => {
                Array::I16(ArrayD::from_shape_vec(sh, ints!(bytes, i16, 2, *be)).map_err(err)?)
            }
            (2, false) => {
                Array::U16(ArrayD::from_shape_vec(sh, ints!(bytes, u16, 2, *be)).map_err(err)?)
            }
            (4, true) => {
                Array::I32(ArrayD::from_shape_vec(sh, ints!(bytes, i32, 4, *be)).map_err(err)?)
            }
            (4, false) => {
                Array::U32(ArrayD::from_shape_vec(sh, ints!(bytes, u32, 4, *be)).map_err(err)?)
            }
            (8, true) => {
                Array::I64(ArrayD::from_shape_vec(sh, ints!(bytes, i64, 8, *be)).map_err(err)?)
            }
            _ => Array::U64(ArrayD::from_shape_vec(sh, ints!(bytes, u64, 8, *be)).map_err(err)?),
        },
        Datatype::Float {
            size: 2,
            big_endian: be,
        } => {
            let halves = ints!(bytes, u16, 2, *be);
            Array::F32(
                ArrayD::from_shape_vec(sh, halves.into_iter().map(crate::half::to_f32).collect())
                    .map_err(err)?,
            )
        }
        Datatype::Float {
            size: 4,
            big_endian: be,
        } => Array::F32(ArrayD::from_shape_vec(sh, ints!(bytes, f32, 4, *be)).map_err(err)?),
        Datatype::Float { big_endian: be, .. } => {
            Array::F64(ArrayD::from_shape_vec(sh, ints!(bytes, f64, 8, *be)).map_err(err)?)
        }
        Datatype::FixedString { size, pad } => {
            let strings: Vec<String> = if *size == 0 {
                vec![String::new(); shape.iter().product()]
            } else {
                bytes.chunks_exact(*size).map(|c| strip(c, *pad)).collect()
            };
            Array::Str(ArrayD::from_shape_vec(sh, strings).map_err(err)?)
        }
        Datatype::VarString { pad } => {
            let esize = dt.element_size(file.sizes.offset);
            let mut heaps: HashMap<u64, HashMap<u16, Vec<u8>>> = HashMap::new();
            let mut strings = Vec::new();
            for el in bytes.chunks_exact(esize) {
                let mut c = Cursor::new(el, 0, file.sizes);
                let len = usize::try_from(c.u32()?).map_err(|_| bad("string too long"))?;
                let addr = c.addr()?;
                let index = u16::try_from(c.u32()?).map_err(|_| bad("bad global heap index"))?;
                let s = match addr {
                    None => String::new(),
                    Some(_) if len == 0 => String::new(),
                    Some(a) => {
                        let heap = match heaps.entry(a) {
                            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(global_heap(file, a)?)
                            }
                        };
                        let obj = heap
                            .get(&index)
                            .ok_or_else(|| bad("missing global heap object"))?;
                        strip(obj.get(..len.min(obj.len())).unwrap_or_default(), *pad)
                    }
                };
                strings.push(s);
            }
            Array::Str(ArrayD::from_shape_vec(sh, strings).map_err(err)?)
        }
        Datatype::Enum { base, .. } => decode(file, base, bytes, shape)?,
        Datatype::PackedInt {
            size,
            signed,
            big_endian,
            offset,
            precision,
        } => {
            let whole = Datatype::Int {
                size: *size,
                signed: *signed,
                big_endian: *big_endian,
            };
            unpack_bits(
                decode(file, &whole, bytes, shape)?,
                *offset,
                *precision,
                *signed,
            )
        }
        Datatype::Other { class, .. } => {
            return Err(Error::Unsupported(format!("HDF5 datatype class {class}")));
        }
        Datatype::ObjectReference | Datatype::ReferenceList => {
            return Err(Error::Unsupported(
                "object references (use Attribute::references)".into(),
            ));
        }
        Datatype::VarSeq { .. } => {
            return Err(Error::Unsupported(
                "variable-length sequences (use Dataset::read_ragged)".into(),
            ));
        }
        Datatype::Compound { .. } => {
            return Err(Error::Unsupported(
                "compound types (use Dataset::read_compound)".into(),
            ));
        }
        Datatype::ArrayOf { .. } => {
            return Err(Error::Unsupported("array types outside compounds".into()));
        }
    })
}

/// A variable-length sequence per element: the lengths (shaped like the dataset) and every
/// element's values, concatenated in C order.
pub(crate) fn decode_ragged(
    file: &File,
    base: &Datatype,
    bytes: &[u8],
    shape: &[usize],
) -> Result<(ArrayD<u64>, Array), Error> {
    let esize = 8usize.saturating_add(usize::from(file.sizes.offset));
    let bsize = base.size();
    let mut heaps: HashMap<u64, HashMap<u16, Vec<u8>>> = HashMap::new();
    let mut counts = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for el in bytes.chunks_exact(esize) {
        let mut c = Cursor::new(el, 0, file.sizes);
        let len = c.u32()?;
        let addr = c.addr()?;
        let index = u16::try_from(c.u32()?).map_err(|_| bad("bad global heap index"))?;
        let n = usize::try_from(len).map_err(|_| bad("sequence too long"))?;
        if let (Some(a), true) = (addr, n > 0) {
            let heap = match heaps.entry(a) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(global_heap(file, a)?),
            };
            let obj = heap
                .get(&index)
                .ok_or_else(|| bad("missing global heap object"))?;
            let want = n
                .checked_mul(bsize)
                .ok_or_else(|| bad("sequence too long"))?;
            values.extend_from_slice(
                obj.get(..want)
                    .ok_or_else(|| bad("sequence shorter than its length"))?,
            );
            counts.push(u64::from(len));
        } else {
            counts.push(0);
        }
    }
    let total = values.len().checked_div(bsize.max(1)).unwrap_or(0);
    let counts = ArrayD::from_shape_vec(IxDyn(shape), counts)
        .map_err(|e| bad(format!("data does not match its shape: {e}")))?;
    Ok((counts, decode(file, base, &values, &[total])?))
}

/// Object references: one address per element (`None` for a null reference).
pub(crate) fn decode_references(file: &File, bytes: &[u8]) -> Result<Vec<Option<u64>>, Error> {
    bytes
        .chunks_exact(usize::from(file.sizes.offset).max(8))
        .map(|c| {
            Cursor::new(c, 0, file.sizes)
                .addr()
                .map(|a| a.filter(|x| *x != 0))
        })
        .collect()
}

/// Variable-length lists of object references, read from the global heap.
pub(crate) fn decode_reference_lists(file: &File, bytes: &[u8]) -> Result<Vec<Vec<u64>>, Error> {
    let esize = 8usize.saturating_add(usize::from(file.sizes.offset));
    let mut heaps: HashMap<u64, HashMap<u16, Vec<u8>>> = HashMap::new();
    let mut out = Vec::new();
    for el in bytes.chunks_exact(esize) {
        let mut c = Cursor::new(el, 0, file.sizes);
        let n = usize::try_from(c.u32()?).map_err(|_| bad("reference list too long"))?;
        let addr = c.addr()?;
        let index = u16::try_from(c.u32()?).map_err(|_| bad("bad global heap index"))?;
        let Some(a) = addr.filter(|_| n > 0) else {
            out.push(Vec::new());
            continue;
        };
        let heap = match heaps.entry(a) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(global_heap(file, a)?),
        };
        let obj = heap
            .get(&index)
            .ok_or_else(|| bad("missing global heap object"))?;
        let refs = decode_references(file, obj)?;
        out.push(refs.into_iter().take(n).flatten().collect());
    }
    Ok(out)
}
