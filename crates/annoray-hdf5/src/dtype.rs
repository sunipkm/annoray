//! HDF5 datatypes and conversion of raw element bytes to typed arrays.

use std::collections::HashMap;

use annoray_core::{DType, Owned, TypedArray};
use annoray_io::IoError;
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
    pub fn value_dtype(&self) -> Option<DType> {
        Some(match self {
            Self::Int {
                size: 1,
                signed: true,
                ..
            } => DType::I8,
            Self::Int { size: 1, .. } => DType::U8,
            Self::Int {
                size: 2,
                signed: true,
                ..
            } => DType::I16,
            Self::Int { size: 2, .. } => DType::U16,
            Self::Int {
                size: 4,
                signed: true,
                ..
            } => DType::I32,
            Self::Int { size: 4, .. } => DType::U32,
            Self::Int {
                size: 8,
                signed: true,
                ..
            } => DType::I64,
            Self::Int { .. } => DType::U64,
            Self::Float { size: 4, .. } => DType::F32,
            Self::Float { .. } => DType::F64,
            Self::FixedString { .. } | Self::VarString { .. } => DType::Str,
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
            Self::ObjectReference | Self::ReferenceList | Self::Other { .. } => return None,
        })
    }

    /// Parse a datatype message.
    pub(crate) fn parse(data: &[u8]) -> Result<Self, IoError> {
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

    fn parse_from(c: &mut Cursor<'_>) -> Result<Self, IoError> {
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
                    Ok(s @ (4 | 8)) => Self::Float {
                        size: s,
                        big_endian: b0 & 1 != 0,
                    },
                    _ => Self::Other { class, size },
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
                    .collect::<Result<Vec<_>, IoError>>()?;
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
                } else {
                    Self::Other { class, size }
                }
            }
            7 if b0 & 0x0F == 0 && size == 8 => Self::ObjectReference,
            2 | 4 | 5 | 6 | 7 | 10 => {
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
            Self::VarString { .. } | Self::ReferenceList => 16,
            Self::ObjectReference => 8,
            Self::Enum { base, .. } => base.size(),
        }
    }

    /// Bytes per element in a file with offset width `offset`.
    pub(crate) fn element_size(&self, offset: u8) -> usize {
        match self {
            Self::VarString { .. } | Self::ReferenceList => {
                8usize.saturating_add(usize::from(offset))
            }
            other => other.size(),
        }
    }
}

/// Resolve a shared (committed) datatype.
pub(crate) fn shared_datatype(file: &File, data: &[u8]) -> Result<Datatype, IoError> {
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
    Datatype::parse(&m.data)
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
fn global_heap(file: &File, addr: u64) -> Result<HashMap<u16, Vec<u8>>, IoError> {
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
    ($bytes:expr, $t:ty, $n:literal, $be:expr) => {
        $bytes
            .chunks_exact($n)
            .map(|c| {
                let a: [u8; $n] = c.try_into().unwrap_or([0; $n]);
                if $be {
                    <$t>::from_be_bytes(a)
                } else {
                    <$t>::from_le_bytes(a)
                }
            })
            .collect::<Vec<$t>>()
    };
}

/// The `precision`-bit field at bit `offset` of every integer, sign-extended if `signed` (what
/// HDF5's conversion of a bit-field integer type to a native one gives).
fn unpack_bits(
    data: TypedArray<Owned>,
    offset: u16,
    precision: u16,
    signed: bool,
) -> TypedArray<Owned> {
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
        TypedArray::I8(a) => TypedArray::I8(go!(a, i8, u8)),
        TypedArray::U8(a) => TypedArray::U8(go!(a, u8, u8)),
        TypedArray::I16(a) => TypedArray::I16(go!(a, i16, u16)),
        TypedArray::U16(a) => TypedArray::U16(go!(a, u16, u16)),
        TypedArray::I32(a) => TypedArray::I32(go!(a, i32, u32)),
        TypedArray::U32(a) => TypedArray::U32(go!(a, u32, u32)),
        TypedArray::I64(a) => TypedArray::I64(go!(a, i64, u64)),
        TypedArray::U64(a) => TypedArray::U64(go!(a, u64, u64)),
        other => other,
    }
}

/// Convert raw element bytes into an array of `shape`.
pub(crate) fn decode(
    file: &File,
    dt: &Datatype,
    bytes: &[u8],
    shape: &[usize],
) -> Result<TypedArray<Owned>, IoError> {
    let sh = IxDyn(shape);
    let err = |e: ndarray::ShapeError| bad(format!("data does not match its shape: {e}"));
    Ok(match dt {
        Datatype::Int {
            size,
            signed,
            big_endian: be,
        } => match (size, signed) {
            (1, true) => {
                TypedArray::I8(ArrayD::from_shape_vec(sh, ints!(bytes, i8, 1, *be)).map_err(err)?)
            }
            (1, false) => TypedArray::U8(ArrayD::from_shape_vec(sh, bytes.to_vec()).map_err(err)?),
            (2, true) => {
                TypedArray::I16(ArrayD::from_shape_vec(sh, ints!(bytes, i16, 2, *be)).map_err(err)?)
            }
            (2, false) => {
                TypedArray::U16(ArrayD::from_shape_vec(sh, ints!(bytes, u16, 2, *be)).map_err(err)?)
            }
            (4, true) => {
                TypedArray::I32(ArrayD::from_shape_vec(sh, ints!(bytes, i32, 4, *be)).map_err(err)?)
            }
            (4, false) => {
                TypedArray::U32(ArrayD::from_shape_vec(sh, ints!(bytes, u32, 4, *be)).map_err(err)?)
            }
            (8, true) => {
                TypedArray::I64(ArrayD::from_shape_vec(sh, ints!(bytes, i64, 8, *be)).map_err(err)?)
            }
            _ => {
                TypedArray::U64(ArrayD::from_shape_vec(sh, ints!(bytes, u64, 8, *be)).map_err(err)?)
            }
        },
        Datatype::Float {
            size: 4,
            big_endian: be,
        } => TypedArray::F32(ArrayD::from_shape_vec(sh, ints!(bytes, f32, 4, *be)).map_err(err)?),
        Datatype::Float { big_endian: be, .. } => {
            TypedArray::F64(ArrayD::from_shape_vec(sh, ints!(bytes, f64, 8, *be)).map_err(err)?)
        }
        Datatype::FixedString { size, pad } => {
            let strings: Vec<String> = if *size == 0 {
                vec![String::new(); shape.iter().product()]
            } else {
                bytes.chunks_exact(*size).map(|c| strip(c, *pad)).collect()
            };
            TypedArray::Str(ArrayD::from_shape_vec(sh, strings).map_err(err)?)
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
            TypedArray::Str(ArrayD::from_shape_vec(sh, strings).map_err(err)?)
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
            return Err(IoError::Unsupported(format!("HDF5 datatype class {class}")));
        }
        Datatype::ObjectReference | Datatype::ReferenceList => {
            return Err(IoError::Unsupported(
                "object references (use Attribute::references)".into(),
            ));
        }
    })
}

/// Object references: one address per element (`None` for a null reference).
pub(crate) fn decode_references(file: &File, bytes: &[u8]) -> Result<Vec<Option<u64>>, IoError> {
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
pub(crate) fn decode_reference_lists(file: &File, bytes: &[u8]) -> Result<Vec<Vec<u64>>, IoError> {
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
