//! Changing files in place: values written into existing datasets and datasets grown along
//! unlimited dimensions, read back by our reader (and by h5py in the Python tests).
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss
)]

use std::path::PathBuf;

use annoray_hdf5::{Array, Editor, File, Node, WDataset, WGroup, WNode};
use ndarray::{ArrayD, IxDyn};

fn grid(rows: usize, from: usize) -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(&[rows, 4]), |ix| ((ix[0] + from) * 10 + ix[1]) as f64)
}

fn sample(name: &str) -> PathBuf {
    let mut u = WDataset::new(Array::F64(grid(3, 0)));
    u.unlimited = vec![true, false];
    u.chunks = Some(vec![2, 4]);
    u.deflate = Some(4);
    u.shuffle = true;
    let mut empty = WDataset::new(Array::I16(ArrayD::zeros(IxDyn(&[0, 3]))));
    empty.unlimited = vec![true, false];
    empty.chunks = Some(vec![5, 3]);
    empty.fill = Some((-1i16).to_le_bytes().to_vec());
    let c = WDataset::new(Array::I32(ndarray::array![1, 2, 3, 4, 5].into_dyn()));
    let mut g = WGroup::default();
    for (n, d) in [("u", u), ("empty", empty), ("c", c)] {
        g.members.push((n.into(), WNode::Dataset(d)));
    }
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-edit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, annoray_hdf5::write_file(&g).unwrap()).unwrap();
    path
}

fn read(path: &PathBuf, name: &str) -> Array {
    let f = File::open(path).unwrap();
    match f.root().unwrap().get(name).unwrap() {
        Node::Dataset(d) => d.read().unwrap(),
        _ => panic!("{name}"),
    }
}

#[test]
fn append_rows_and_write_in_place() {
    let path = sample("append.h5");
    let mut e = Editor::open(&path).unwrap();
    assert_eq!(e.shape("u").unwrap(), vec![3, 4]);
    assert_eq!(e.max_shape("u").unwrap(), vec![None, Some(4)]);
    e.resize("u", &[7, 4]).unwrap();
    e.write("u", &[3, 0], &Array::F64(grid(4, 3))).unwrap();
    // A partial chunk of existing data.
    let one = ArrayD::from_elem(IxDyn(&[1, 2]), -5.0);
    e.write("u", &[0, 1], &Array::F64(one)).unwrap();
    e.write("c", &[1], &Array::I32(ndarray::array![20, 30].into_dyn()))
        .unwrap();
    e.resize("empty", &[2, 3]).unwrap();
    e.write(
        "empty",
        &[1, 0],
        &Array::I16(ndarray::array![[7i16, 8, 9]].into_dyn()),
    )
    .unwrap();
    // Seen before flushing.
    let Array::F64(before) = e.read("u", &[5, 0], &[2, 4]).unwrap() else {
        panic!()
    };
    assert_eq!(before, grid(2, 5));
    e.close().unwrap();

    let mut want = grid(7, 0);
    want[[0, 1]] = -5.0;
    want[[0, 2]] = -5.0;
    assert_eq!(read(&path, "u"), Array::F64(want));
    assert_eq!(
        read(&path, "c"),
        Array::I32(ndarray::array![1, 20, 30, 4, 5].into_dyn())
    );
    assert_eq!(
        read(&path, "empty"),
        Array::I16(ndarray::array![[-1i16, -1, -1], [7, 8, 9]].into_dyn())
    );
}

#[test]
fn many_appends_build_a_deep_index() {
    let path = sample("deep.h5");
    let mut e = Editor::open(&path).unwrap();
    // 3 rows, then 1 row at a time: hundreds of chunks, several B-tree levels.
    let mut rows = 3;
    for step in 0..300 {
        e.resize("u", &[rows + 1, 4]).unwrap();
        e.write("u", &[rows, 0], &Array::F64(grid(1, rows)))
            .unwrap();
        rows += 1;
        if step % 97 == 0 {
            e.flush().unwrap();
        }
    }
    e.close().unwrap();
    assert_eq!(read(&path, "u"), Array::F64(grid(rows, 0)));
}

#[test]
fn errors() {
    let path = sample("errors.h5");
    let mut e = Editor::open(&path).unwrap();
    assert!(e.resize("u", &[2, 4]).is_err(), "shrinking");
    assert!(e.resize("u", &[3, 5]).is_err(), "past the maximum");
    assert!(e.resize("c", &[6]).is_err(), "contiguous");
    assert!(
        e.write("u", &[2, 0], &Array::F64(grid(2, 0))).is_err(),
        "outside"
    );
    let ints = Array::I32(ArrayD::zeros(IxDyn(&[1, 4])));
    assert!(e.write("u", &[0, 0], &ints).is_err(), "another type");
}
