//! Files written by the pure-Rust writer must read back identically with our reader and —
//! when Python is available — with h5py. (`annoray`'s tests check netCDF-C reads them too.)
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray_hdf5::{Array, AttrData, File, Node, WDataset, WGroup, WNode};
use ndarray::{ArrayD, IxDyn, array};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-w-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn sample() -> WGroup<'static> {
    let mut chunked = WDataset::new(Array::F64(ArrayD::from_shape_fn(IxDyn(&[7, 5]), |ix| {
        (ix[0] * 10 + ix[1]) as f64 / 3.0
    })));
    chunked.chunks = Some(vec![3, 2]);
    chunked.deflate = Some(5);
    chunked.shuffle = true;
    chunked
        .attrs
        .push(("units".into(), AttrData::Text("m s-1".into())));
    let mut unlimited = WDataset::new(Array::I32(array![1, 2, 3, 4, 5].into_dyn()));
    unlimited.unlimited = vec![true];
    unlimited.chunks = Some(vec![2]);
    let mut filled = WDataset::new(Array::I16(array![-1i16, 2, 3].into_dyn()));
    filled.fill = Some((-999i16).to_le_bytes().to_vec());
    let mut chars = WDataset::new(Array::U8(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), b"abxyz\0".to_vec()).unwrap(),
    ));
    chars.char_data = true;
    let mut referencing = WDataset::new(Array::F32(array![1.5f32].into_dyn()));
    referencing.attrs.push((
        "refs".into(),
        AttrData::References(vec!["/contig".into(), "/group/inner".into()]),
    ));
    referencing.attrs.push((
        "ref_lists".into(),
        AttrData::ReferenceLists(vec![vec!["/contig".into()]]),
    ));
    referencing.attrs.push((
        "pairs".into(),
        AttrData::ReferenceIndexPairs(vec![("/contig".into(), 0), ("/chunked".into(), 1)]),
    ));
    let mut many = WGroup::default();
    for i in 0..40 {
        many.members.push((
            format!("d{i:02}"),
            WNode::Dataset(WDataset::new(Array::I64(ArrayD::from_elem(IxDyn(&[]), i)))),
        ));
    }
    let mut inner = WGroup::default();
    inner.attrs.push((
        "level".into(),
        AttrData::Array(Array::I32(array![1].into_dyn())),
    ));
    inner.members.push((
        "inner".into(),
        WNode::Dataset(WDataset::new(Array::U64(array![u64::MAX, 7].into_dyn()))),
    ));
    WGroup {
        attrs: vec![
            ("title".into(), AttrData::Text("written by annoray".into())),
            (
                "ints".into(),
                AttrData::Array(Array::I32(array![1, -2, 3].into_dyn())),
            ),
            (
                "doubles".into(),
                AttrData::Array(Array::F64(array![0.5].into_dyn())),
            ),
            (
                "vlen".into(),
                AttrData::Array(Array::Str(
                    array!["α".to_owned(), String::new(), "βγ".to_owned()].into_dyn(),
                )),
            ),
            ("vstr".into(), AttrData::VarString("scalar vlen".into())),
            ("empty".into(), AttrData::Text("".into())),
        ],
        members: vec![
            (
                "contig".into(),
                WNode::Dataset(WDataset::new(Array::U8(array![1u8, 2, 255].into_dyn()))),
            ),
            ("chunked".into(), WNode::Dataset(chunked)),
            ("unlimited".into(), WNode::Dataset(unlimited)),
            ("filled".into(), WNode::Dataset(filled)),
            (
                "scalar".into(),
                WNode::Dataset(WDataset::new(Array::F32(ArrayD::from_elem(
                    IxDyn(&[]),
                    2.5,
                )))),
            ),
            (
                "empty_ds".into(),
                WNode::Dataset(WDataset::new(Array::F64(ArrayD::zeros(IxDyn(&[0, 4]))))),
            ),
            (
                "strings".into(),
                WNode::Dataset(WDataset::new(Array::Str(
                    array!["héllo".to_owned(), String::new(), "x".repeat(300)].into_dyn(),
                ))),
            ),
            ("chars".into(), WNode::Dataset(chars)),
            ("referencing".into(), WNode::Dataset(referencing)),
            ("group".into(), WNode::Group(inner)),
            ("many".into(), WNode::Group(many)),
        ],
    }
}

/// Write the sample to a file of its own (tests run in parallel).
fn written(name: &str) -> PathBuf {
    let path = tmp(name);
    std::fs::write(&path, annoray_hdf5::write_file(&sample()).unwrap()).unwrap();
    path
}

#[test]
fn our_reader_reads_our_files() {
    let f = File::open(written("ours.h5")).unwrap();
    let root = f.root().unwrap();
    let names: Vec<String> = root
        .members()
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        names,
        [
            "contig",
            "chunked",
            "unlimited",
            "filled",
            "scalar",
            "empty_ds",
            "strings",
            "chars",
            "referencing",
            "group",
            "many"
        ]
    );
    let get = |p: &str| match root.get(p).unwrap() {
        Node::Dataset(d) => d,
        _ => panic!("{p}"),
    };
    let chunked = get("chunked");
    assert_eq!(chunked.chunks(), Some(vec![3, 2]));
    assert_eq!(chunked.compression(), (Some(5), true));
    let Array::F64(v) = chunked.read().unwrap() else {
        panic!()
    };
    assert_eq!(v[[6, 4]], 64.0 / 3.0);
    assert_eq!(get("unlimited").max_shape(), vec![None]);
    assert!(
        matches!(get("strings").read().unwrap(), Array::Str(s) if s[[2]].len() == 300 && s[[0]] == "héllo")
    );
    assert!(matches!(get("chars").read().unwrap(), Array::Str(s) if s.shape() == [2, 3]));
    assert_eq!(get("empty_ds").shape(), vec![0, 4]);
    assert!(matches!(get("group/inner").read().unwrap(), Array::U64(x) if x[[0]] == u64::MAX));
    let attrs: Vec<String> = root.attrs().unwrap().into_iter().map(|a| a.name).collect();
    assert_eq!(attrs, ["title", "ints", "doubles", "vlen", "vstr", "empty"]);
    let vlen = root
        .attrs()
        .unwrap()
        .into_iter()
        .find(|a| a.name == "vlen")
        .unwrap()
        .value()
        .unwrap();
    assert!(
        matches!(vlen, Array::Str(s) if s.iter().cloned().collect::<Vec<_>>() == ["α", "", "βγ"])
    );
    match root.get("many").unwrap() {
        Node::Group(g) => assert_eq!(g.members().unwrap().len(), 40),
        _ => panic!(),
    }
}

/// h5py (the HDF5 C library) must see exactly what our reader sees.
/// `cargo test -p annoray-hdf5 --test write_round_trip -- --ignored`
#[test]
#[ignore = "needs python3 with h5py"]
fn h5py_reads_our_files() {
    let path = written("py.h5");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/h5py_dump.py");
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&script)
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let dump = String::from_utf8(out.stdout).unwrap();
    for needle in [
        "A /@title str scalar \"written by annoray\"",
        "A /@vlen str 3 \"\\u03b1\" \"\" \"\\u03b2\\u03b3\"",
        "D /chunked f64 7x5 0.0 0.3333333333333333",
        "D /unlimited i32 5 1 2 3 4 5",
        "D /filled i16 3 -1 2 3",
        "D /scalar f32 scalar 2.5",
        "D /empty_ds f64 0x4",
        "D /group/inner u64 2 18446744073709551615 7",
        "D /many/d39 i64 scalar 39",
        "A /chunked@units str scalar \"m s-1\"",
        // References resolve to the right objects, in every attribute form.
        "R /referencing@refs /contig /group/inner",
        "R /referencing@ref_lists [/contig]",
        "R /referencing@pairs (/contig,0) (/chunked,1)",
    ] {
        assert!(dump.contains(needle), "missing {needle:?} in\n{dump}");
    }
}
