//! Virtual datasets (layout class 3): values gathered from other datasets, in this file or
//! others, by mappings from a selection of each source to a selection of the virtual
//! dataset. The mappings are a global heap object (`H5D__virtual_load_layout`): a version, a
//! count, and per mapping the source file (`.` for this one) and dataset names, the source
//! selection and the virtual selection, then a checksum. Elements map in order: the k-th
//! selected element of the source (row-major for hyperslabs, listed order for points) goes
//! to the k-th of the virtual selection. Elements no mapping covers hold the fill value.
//!
//! Unlimited mappings and `printf`-style source names (`%b`) are not supported; nor is type
//! conversion: a source must store the virtual dataset's own type.

use std::cell::Cell;
use std::path::Path;

use crate::checksum::lookup3;
use crate::ck::Ck;
use crate::cursor::Cursor;
use crate::{Dataset, Error, File, Node, bad};

/// Virtual datasets whose sources are virtual, nested at most this deep (a cycle otherwise).
const MAX_DEPTH: usize = 8;

/// Selections expanded to more boxes than this are refused (a corrupt or hostile mapping).
const MAX_BOXES: u64 = 1 << 22;

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// A dataspace selection.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selection {
    All,
    None,
    /// Points, in their order.
    Points(Vec<Vec<u64>>),
    /// Hyperslab blocks: `(start, count)` per block.
    Boxes(Vec<(Vec<u64>, Vec<u64>)>),
}

#[derive(Debug)]
struct Mapping {
    file: String,
    dataset: String,
    source: Selection,
    target: Selection,
}

fn uint(c: &mut Cursor<'_>, size: u8) -> Result<u64, Error> {
    match size {
        2 => c.u16().map(u64::from),
        4 => c.u32().map(u64::from),
        8 => c.u64(),
        _ => Err(bad(format!("selection field of {size} bytes"))),
    }
}

/// `true` for the encoded "unlimited" of a `size`-byte field.
fn unlimited(v: u64, size: u8) -> bool {
    match size {
        2 => v == u64::from(u16::MAX),
        4 => v == u64::from(u32::MAX),
        _ => v == u64::MAX,
    }
}

/// A serialized selection (`H5S_select_deserialize`).
fn selection(c: &mut Cursor<'_>) -> Result<Selection, Error> {
    let kind = c.u32()?;
    let version = c.u32()?;
    match kind {
        // None / all: version, reserved, length.
        0 | 3 => {
            c.skip(8)?;
            Ok(if kind == 0 {
                Selection::None
            } else {
                Selection::All
            })
        }
        1 => {
            let size = if version >= 2 {
                c.u8()?
            } else {
                c.skip(8)?;
                4
            };
            let rank = usize::try_from(c.u32()?).map_err(|_| bad("selection rank"))?;
            if rank == 0 || rank > 32 {
                return Err(bad("selection rank"));
            }
            let n = uint(c, size)?;
            if n > MAX_BOXES {
                return Err(bad("too many points in a selection"));
            }
            (0..n)
                .map(|_| (0..rank).map(|_| uint(c, size)).collect())
                .collect::<Result<Vec<_>, _>>()
                .map(Selection::Points)
        }
        2 => {
            let (flags, size) = if version >= 3 {
                (c.u8()?, c.u8()?)
            } else if version == 2 {
                let f = c.u8()?;
                c.skip(4)?;
                (f, 8)
            } else {
                c.skip(8)?;
                (0, 4)
            };
            let rank = usize::try_from(c.u32()?).map_err(|_| bad("selection rank"))?;
            if rank == 0 || rank > 32 {
                return Err(bad("selection rank"));
            }
            if flags & 1 != 0 {
                // Regular: start, stride, count, block per dimension.
                let mut dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    let start = uint(c, size)?;
                    let stride = uint(c, size)?;
                    let count = uint(c, size)?;
                    let block = uint(c, size)?;
                    if unlimited(count, size) || unlimited(block, size) {
                        return Err(Error::Unsupported(
                            "unlimited virtual dataset mappings".into(),
                        ));
                    }
                    dims.push((start, stride, count, block));
                }
                let total = dims
                    .iter()
                    .try_fold(1u64, |a, d| a.checked_mul(d.2))
                    .filter(|t| *t <= MAX_BOXES)
                    .ok_or_else(|| bad("too many blocks in a selection"))?;
                let counts: Vec<u64> = dims.iter().map(|d| d.2).collect();
                let mut boxes = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
                for k in 0..total {
                    let idx = crate::unravel(k, &counts).ok_or_else(|| bad("selection"))?;
                    let start = dims
                        .iter()
                        .zip(&idx)
                        .map(|(d, i)| {
                            i.checked_mul(d.1)
                                .and_then(|o| o.checked_add(d.0))
                                .ok_or_else(|| bad("selection offset"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    boxes.push((start, dims.iter().map(|d| d.3).collect()));
                }
                Ok(Selection::Boxes(boxes))
            } else {
                let n = uint(c, size)?;
                if n > MAX_BOXES {
                    return Err(bad("too many blocks in a selection"));
                }
                let mut boxes = Vec::new();
                for _ in 0..n {
                    let start = (0..rank)
                        .map(|_| uint(c, size))
                        .collect::<Result<Vec<_>, _>>()?;
                    let end = (0..rank)
                        .map(|_| uint(c, size))
                        .collect::<Result<Vec<_>, _>>()?;
                    let count = start
                        .iter()
                        .zip(&end)
                        .map(|(s, e)| {
                            e.checked_sub(*s)
                                .and_then(|x| x.checked_add(1))
                                .ok_or_else(|| bad("hyperslab block ends before it starts"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    boxes.push((start, count));
                }
                Ok(Selection::Boxes(boxes))
            }
        }
        k => Err(bad(format!("selection type {k}"))),
    }
}

fn c_string(c: &mut Cursor<'_>) -> Result<String, Error> {
    let rest = c.rest();
    let n = rest
        .iter()
        .position(|b| *b == 0)
        .ok_or_else(|| bad("unterminated name in virtual dataset mappings"))?;
    let s = String::from_utf8_lossy(rest.get(..n).unwrap_or_default()).into_owned();
    c.skip(n.saturating_add(1))?;
    Ok(s)
}

/// The mappings of a virtual dataset (its global heap object `index` at `heap`).
fn mappings(file: &File, heap: u64, index: u32) -> Result<Vec<Mapping>, Error> {
    let objects = crate::dtype::global_heap(file, heap)?;
    let block = u16::try_from(index)
        .ok()
        .and_then(|i| objects.get(&i))
        .ok_or_else(|| bad("virtual dataset mappings not in the global heap"))?;
    let body_len = block
        .len()
        .checked_sub(4)
        .ok_or_else(|| bad("virtual dataset mappings truncated"))?;
    let (body, sum) = block.split_at(body_len);
    if lookup3(body, 0).to_le_bytes() != sum {
        return Err(bad("virtual dataset mappings: checksum mismatch"));
    }
    let mut c = Cursor::new(body, 0, file.sizes);
    let version = c.u8()?;
    if version > 1 {
        return Err(bad(format!("virtual dataset mappings version {version}")));
    }
    let n = c.length()?;
    if n > MAX_BOXES {
        return Err(bad("too many virtual dataset mappings"));
    }
    let mut out: Vec<Mapping> = Vec::new();
    for i in 0..n {
        let flags = if version >= 1 { c.u8()? } else { 0 };
        // Names may be shared with an earlier mapping (by its index).
        let earlier = |c: &mut Cursor<'_>, out: &[Mapping]| -> Result<usize, Error> {
            let k = usize::try_from(c.length()?).map_err(|_| bad("mapping index"))?;
            if (k as u64) >= i || k >= out.len() {
                return Err(bad("a mapping shares a name with a later one"));
            }
            Ok(k)
        };
        let file_name = if flags & 0x04 != 0 {
            ".".to_owned()
        } else if flags & 0x01 != 0 {
            let k = earlier(&mut c, &out)?;
            out.get(k).map(|m| m.file.clone()).unwrap_or_default()
        } else {
            c_string(&mut c)?
        };
        let dataset = if flags & 0x02 != 0 {
            let k = earlier(&mut c, &out)?;
            out.get(k).map(|m| m.dataset.clone()).unwrap_or_default()
        } else {
            c_string(&mut c)?
        };
        let source = selection(&mut c)?;
        let target = selection(&mut c)?;
        if file_name.contains('%') || dataset.contains('%') {
            return Err(Error::Unsupported(
                "virtual dataset source names with printf-style patterns".into(),
            ));
        }
        out.push(Mapping {
            file: file_name,
            dataset,
            source,
            target,
        });
    }
    Ok(out)
}

/// The source dataset a mapping names.
fn source(vds: &Dataset, m: &Mapping) -> Result<Dataset, Error> {
    let file = if m.file == "." {
        vds.file.clone()
    } else {
        let p = Path::new(&m.file);
        let full = match (&vds.file.dir, p.is_absolute()) {
            (Some(dir), false) => dir.join(p),
            _ => p.to_path_buf(),
        };
        File::open(&full).map_err(|e| bad(format!("virtual dataset source {:?}: {e}", m.file)))?
    };
    match file.root()?.get(&m.dataset)? {
        Node::Dataset(d) => Ok(*d),
        _ => Err(bad(format!(
            "virtual dataset source {:?} is not a dataset",
            m.dataset
        ))),
    }
}

/// A selection as boxes over `shape` (`All` is one box; points are boxes of one element).
fn boxes(sel: &Selection, shape: &[u64]) -> Vec<(Vec<u64>, Vec<u64>)> {
    match sel {
        Selection::All => vec![(vec![0; shape.len()], shape.to_vec())],
        Selection::None => Vec::new(),
        Selection::Points(p) => p.iter().map(|x| (x.clone(), vec![1; x.len()])).collect(),
        Selection::Boxes(b) => b.clone(),
    }
}

/// The extents of a box other than 1, in order (two boxes with the same such extents list
/// their elements in the same row-major order).
fn squeezed(count: &[u64]) -> Vec<u64> {
    count.iter().copied().filter(|c| *c != 1).collect()
}

/// Every selected element's row-major index over `shape`: in row-major order for
/// hyperslabs, in their order for points.
fn elements(sel: &Selection, shape: &[u64]) -> Result<Vec<u64>, Error> {
    let flat = |p: &[u64]| -> Option<u64> {
        p.iter().zip(shape).try_fold(0u64, |a, (x, d)| {
            if x >= d {
                return None;
            }
            a.checked_mul(*d)?.checked_add(*x)
        })
    };
    let mut out = Vec::new();
    for (start, count) in boxes(sel, shape) {
        let n = count
            .iter()
            .try_fold(1u64, |a, c| a.checked_mul(*c))
            .ok_or_else(|| bad("selection too large"))?;
        for k in 0..n {
            let idx = crate::unravel(k, &count).ok_or_else(|| bad("selection"))?;
            let p: Vec<u64> = start
                .iter()
                .zip(&idx)
                .map(|(s, i)| s.saturating_add(*i))
                .collect();
            out.push(flat(&p).ok_or_else(|| bad("selection outside the dataspace"))?);
        }
    }
    if !matches!(sel, Selection::Points(_)) {
        out.sort_unstable();
        out.dedup();
    }
    Ok(out)
}

fn to_usize(v: &[u64]) -> Result<Vec<usize>, Error> {
    v.iter()
        .map(|x| usize::try_from(*x).map_err(|_| bad("selection too large")))
        .collect()
}

/// Copy the box `[start, start + count)` of virtual dataset `vds` (of `shape`) into `out`
/// (C order, `esize` bytes per element, holding the fill value).
#[allow(clippy::too_many_arguments)]
pub(crate) fn place(
    vds: &Dataset,
    heap: u64,
    index: u32,
    shape: &[usize],
    start: &[usize],
    count: &[usize],
    esize: usize,
    out: &mut [u8],
) -> Result<(), Error> {
    let depth = DEPTH.with(Cell::get);
    if depth >= MAX_DEPTH {
        return Err(bad("virtual datasets nest too deeply (a cycle?)"));
    }
    DEPTH.with(|d| d.set(depth.saturating_add(1)));
    let result = place_mappings(vds, heap, index, shape, start, count, esize, out);
    DEPTH.with(|d| d.set(depth));
    result
}

#[allow(clippy::too_many_arguments)]
fn place_mappings(
    vds: &Dataset,
    heap: u64,
    index: u32,
    shape: &[usize],
    start: &[usize],
    count: &[usize],
    esize: usize,
    out: &mut [u8],
) -> Result<(), Error> {
    let vshape: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
    for m in mappings(&vds.file, heap, index)? {
        let src = source(vds, &m)?;
        if src.dtype != vds.dtype {
            return Err(Error::Unsupported(format!(
                "virtual dataset {}: source {:?} has another type (conversion)",
                vds.path, m.dataset
            )));
        }
        let sshape_us = src.shape();
        let sshape: Vec<u64> = sshape_us.iter().map(|d| *d as u64).collect();
        let sboxes = boxes(&m.source, &sshape);
        let vboxes = boxes(&m.target, &vshape);
        match (sboxes.as_slice(), vboxes.as_slice()) {
            // One box to one box of the same extents: copied box by box.
            ([(s0, sc)], [(v0, vc)]) if squeezed(sc) == squeezed(vc) => {
                place_box_mapping(&src, &sshape_us, s0, sc, v0, vc, start, count, esize, out)?;
            }
            _ => place_elements(&src, &m, &sshape, &vshape, start, count, esize, out)?,
        }
    }
    Ok(())
}

/// One source box onto one virtual box with the same extents (other than 1s): the part
/// inside the requested box is read from the source as a box and copied.
#[allow(clippy::too_many_arguments)]
fn place_box_mapping(
    src: &Dataset,
    sshape: &[usize],
    s0: &[u64],
    sc: &[u64],
    v0: &[u64],
    vc: &[u64],
    start: &[usize],
    count: &[usize],
    esize: usize,
    out: &mut [u8],
) -> Result<(), Error> {
    // The part of the virtual box inside the requested box.
    let mut from = Vec::with_capacity(v0.len());
    let mut to = Vec::with_capacity(v0.len());
    for k in 0..v0.len() {
        let (b0, bc) = (
            v0.get(k).copied().unwrap_or(0),
            vc.get(k).copied().unwrap_or(0),
        );
        let (r0, rc) = (
            start.get(k).map_or(0, |x| *x as u64),
            count.get(k).map_or(0, |x| *x as u64),
        );
        let lo = b0.max(r0);
        let hi = b0.saturating_add(bc).min(r0.saturating_add(rc));
        if lo >= hi {
            return Ok(());
        }
        from.push(lo);
        to.push(hi);
    }
    // The matching source box: the virtual box's non-1 axes in order onto the source's.
    let mut s_start = s0.to_vec();
    let mut s_count = sc.to_vec();
    let src_axes: Vec<usize> = (0..sc.len()).filter(|k| sc.get(*k) != Some(&1)).collect();
    let mut next = src_axes.iter();
    for k in 0..vc.len() {
        if vc.get(k) == Some(&1) {
            continue;
        }
        let axis = *next.next().ok_or_else(|| bad("virtual mapping extents"))?;
        let off = from
            .get(k)
            .copied()
            .unwrap_or(0)
            .saturating_sub(v0.get(k).copied().unwrap_or(0));
        let len = to
            .get(k)
            .copied()
            .unwrap_or(0)
            .saturating_sub(from.get(k).copied().unwrap_or(0));
        if let (Some(s), Some(c)) = (s_start.get_mut(axis), s_count.get_mut(axis)) {
            *s = s
                .checked_add(off)
                .ok_or_else(|| bad("virtual mapping offset"))?;
            *c = len;
        }
    }
    let s_start = to_usize(&s_start)?;
    let s_count = to_usize(&s_count)?;
    if s_start
        .iter()
        .zip(&s_count)
        .zip(sshape)
        .any(|((s, c), d)| s.saturating_add(*c) > *d)
    {
        return Err(bad("virtual mapping outside its source"));
    }
    let n = s_count
        .iter()
        .try_fold(Ck::from(1usize), |a, c| Some(a * *c))
        .and_then(Ck::usize)
        .ok_or_else(|| bad("virtual mapping too large"))?;
    let mut tmp = crate::filled(&src.fill_bytes(esize), n);
    src.place_box(sshape, &s_start, &s_count, esize, &mut tmp)?;
    let from = to_usize(&from)?;
    let to = to_usize(&to)?;
    let dims: Vec<usize> = from
        .iter()
        .zip(&to)
        .map(|(a, b)| b.saturating_sub(*a))
        .collect();
    crate::copy_box(&tmp, &dims, &from, out, count, start, &from, &to, esize)
}

/// Any other mapping: element by element, from the source selection's bounding box.
#[allow(clippy::too_many_arguments)]
fn place_elements(
    src: &Dataset,
    m: &Mapping,
    sshape: &[u64],
    vshape: &[u64],
    start: &[usize],
    count: &[usize],
    esize: usize,
    out: &mut [u8],
) -> Result<(), Error> {
    let s_el = elements(&m.source, sshape)?;
    let v_el = elements(&m.target, vshape)?;
    if s_el.len() != v_el.len() {
        return Err(bad(format!(
            "virtual mapping of {:?}: {} source elements for {} virtual ones",
            m.dataset,
            s_el.len(),
            v_el.len()
        )));
    }
    if s_el.is_empty() {
        return Ok(());
    }
    // The source's bounding box of the selection.
    let mut lo = vec![u64::MAX; sshape.len()];
    let mut hi = vec![0u64; sshape.len()];
    for f in &s_el {
        let p = crate::unravel(*f, sshape).ok_or_else(|| bad("selection"))?;
        for (k, x) in p.iter().enumerate() {
            if let (Some(l), Some(h)) = (lo.get_mut(k), hi.get_mut(k)) {
                *l = (*l).min(*x);
                *h = (*h).max(x.saturating_add(1));
            }
        }
    }
    let b_start = to_usize(&lo)?;
    let b_count: Vec<usize> = to_usize(&hi)?
        .iter()
        .zip(&b_start)
        .map(|(h, l)| h.saturating_sub(*l))
        .collect();
    let n = b_count
        .iter()
        .try_fold(Ck::from(1usize), |a, c| Some(a * *c))
        .and_then(Ck::usize)
        .ok_or_else(|| bad("virtual mapping too large"))?;
    let mut tmp = crate::filled(&src.fill_bytes(esize), n);
    src.place_box(&to_usize(sshape)?, &b_start, &b_count, esize, &mut tmp)?;
    let b_dims: Vec<u64> = b_count.iter().map(|x| *x as u64).collect();
    let r_dims: Vec<u64> = count.iter().map(|x| *x as u64).collect();
    for (s, v) in s_el.iter().zip(&v_el) {
        let vp = crate::unravel(*v, vshape).ok_or_else(|| bad("selection"))?;
        // Only elements inside the requested box.
        let inside = vp.iter().zip(start).zip(count).all(|((x, s0), c)| {
            let s0 = *s0 as u64;
            *x >= s0 && *x < s0.saturating_add(*c as u64)
        });
        if !inside {
            continue;
        }
        let sp = crate::unravel(*s, sshape).ok_or_else(|| bad("selection"))?;
        let si = sp
            .iter()
            .zip(&lo)
            .zip(&b_dims)
            .try_fold(0u64, |a, ((x, l), d)| {
                a.checked_mul(*d)?.checked_add(x.checked_sub(*l)?)
            })
            .ok_or_else(|| bad("selection"))?;
        let di = vp
            .iter()
            .zip(start)
            .zip(&r_dims)
            .try_fold(0u64, |a, ((x, s0), d)| {
                a.checked_mul(*d)?.checked_add(x.checked_sub(*s0 as u64)?)
            })
            .ok_or_else(|| bad("selection"))?;
        let si = usize::try_from(si).ok().and_then(|i| i.checked_mul(esize));
        let di = usize::try_from(di).ok().and_then(|i| i.checked_mul(esize));
        let (Some(si), Some(di)) = (si, di) else {
            return Err(bad("selection offset"));
        };
        let (Some(d), Some(s)) = (
            out.get_mut(di..di.saturating_add(esize)),
            tmp.get(si..si.saturating_add(esize)),
        ) else {
            return Err(bad("selection outside its bounds"));
        };
        d.copy_from_slice(s);
    }
    Ok(())
}
