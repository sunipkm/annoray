//! Writer options: the Fletcher-32 checksum filter and big-endian storage, read back by our
//! reader and (when Python is available) by h5py.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray_hdf5::{Array, File, Node, WDataset, WGroup, WNode};
use ndarray::{ArrayD, IxDyn, array};

fn sample() -> WGroup<'static> {
    let values = || ArrayD::from_shape_fn(IxDyn(&[6, 5]), |ix| (ix[0] * 10 + ix[1]) as f64 + 0.25);
    let mut checked = WDataset::new(Array::F64(values()));
    checked.fletcher32 = true;
    checked.chunks = Some(vec![4, 2]);
    let mut checked_deflate = WDataset::new(Array::I32(array![1, 2, 3, 4, 5, 6, 7].into_dyn()));
    checked_deflate.fletcher32 = true;
    checked_deflate.deflate = Some(4);
    checked_deflate.shuffle = true; // chunks chosen by the writer
    let mut be = WDataset::new(Array::F64(values()));
    be.big_endian = true;
    let mut be_chunked = WDataset::new(Array::I16(array![-3i16, 7, 300, -32000].into_dyn()));
    be_chunked.big_endian = true;
    be_chunked.chunks = Some(vec![3]);
    be_chunked.deflate = Some(1);
    be_chunked.fill = Some((-999i16).to_le_bytes().to_vec());
    let mut be_u64 = WDataset::new(Array::U64(array![1u64, u64::MAX].into_dyn()));
    be_u64.big_endian = true;
    let mut g = WGroup::default();
    for (n, d) in [
        ("checked", checked),
        ("checked_deflate", checked_deflate),
        ("be", be),
        ("be_chunked", be_chunked),
        ("be_u64", be_u64),
    ] {
        g.members.push((n.into(), WNode::Dataset(d)));
    }
    g
}

fn written(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-opt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, annoray_hdf5::write_file(&sample()).unwrap()).unwrap();
    path
}

#[test]
fn checksums_and_byte_order_read_back() {
    let f = File::open(written("ours.h5")).unwrap();
    let root = f.root().unwrap();
    let get = |p: &str| match root.get(p).unwrap() {
        Node::Dataset(d) => d,
        _ => panic!("{p}"),
    };
    let c = get("checked");
    assert!(c.fletcher32());
    assert_eq!(c.chunks(), Some(vec![4, 2]));
    assert!(matches!(c.read().unwrap(), Array::F64(v) if v[[5, 4]] == 54.25));
    let cd = get("checked_deflate");
    assert!(cd.fletcher32() && cd.chunks().is_some() && cd.compression() == (Some(4), true));
    assert!(matches!(cd.read().unwrap(), Array::I32(v) if v.iter().copied().eq(1..=7)));
    let be = get("be");
    assert!(be.big_endian() && be.contiguous());
    assert!(matches!(be.read().unwrap(), Array::F64(v) if v[[3, 2]] == 32.25));
    assert!(
        matches!(be.read_slice(&[1, 1], &[2, 2]).unwrap(), Array::F64(v) if v[[1, 1]] == 22.25)
    );
    let bc = get("be_chunked");
    assert!(bc.big_endian());
    assert!(
        matches!(bc.read().unwrap(), Array::I16(v) if v.iter().copied().eq([-3, 7, 300, -32000]))
    );
    assert_eq!(bc.fill_value(), Some(&(-999i16).to_be_bytes()[..]));
    assert!(matches!(get("be_u64").read().unwrap(), Array::U64(v) if v[[1]] == u64::MAX));
}

#[test]
fn a_corrupted_checksummed_chunk_is_an_error() {
    let path = written("corrupt.h5");
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip a byte of the last chunk's data (chunks are written after the metadata).
    let target = 54.25f64.to_le_bytes();
    let at = bytes
        .windows(8)
        .rposition(|w| w == target)
        .expect("the value is stored");
    bytes[at] ^= 0x40;
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let Node::Dataset(d) = f.root().unwrap().get("checked").unwrap() else {
        panic!()
    };
    let e = d.read().unwrap_err().to_string();
    assert!(e.contains("fletcher32"), "{e}");
}

/// h5py (the HDF5 C library) must read the same values and see the same filters.
#[test]
#[ignore = "needs python3 with h5py"]
fn h5py_reads_checksums_and_byte_order() {
    let path = written("py.h5");
    let script = r#"
import sys, h5py, numpy as np
with h5py.File(sys.argv[1]) as f:
    c = f["checked"]
    assert c.fletcher32 and c.chunks == (4, 2), (c.fletcher32, c.chunks)
    assert c[5, 4] == 54.25
    cd = f["checked_deflate"]
    assert cd.fletcher32 and cd.compression == "gzip" and cd.shuffle
    assert list(cd[:]) == list(range(1, 8))
    be = f["be"]
    assert be.dtype.byteorder == ">" and be[3, 2] == 32.25, be.dtype
    bc = f["be_chunked"]
    assert bc.dtype.byteorder == ">" and list(bc[:]) == [-3, 7, 300, -32000]
    assert bc.fillvalue == -999
    assert f["be_u64"].dtype.byteorder == ">" and f["be_u64"][1] == np.iinfo(np.uint64).max
print("ok")
"#;
    let out = std::process::Command::new("python3")
        .args(["-I", "-c", script])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
