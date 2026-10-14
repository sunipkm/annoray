//! The pure-Rust reader must reproduce h5py's view of every fixture exactly
//! (`fixtures/gen_hdf5.py` writes the files and their canonical dumps).
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use proptest::prelude::*;
use xr_core::{Owned, TypedArray};
use xr_hdf5::{File, Group, Node};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/h5")
}

/// Python's `repr(float(v))`.
fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sci = format!("{v:e}");
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if v != 0.0 && !(-4..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    } else {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    }
}

/// Python's `json.dumps(s)` (ASCII-only output).
fn py_json(s: &str) -> String {
    let mut out = String::from('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn dump_array(a: &TypedArray<Owned>, scalar: bool) -> String {
    let shape = if scalar {
        "scalar".to_string()
    } else {
        a.shape()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x")
    };
    let (dtype, vals): (&str, Vec<String>) = match a {
        TypedArray::I8(x) => ("i8", x.iter().map(i8::to_string).collect()),
        TypedArray::I16(x) => ("i16", x.iter().map(i16::to_string).collect()),
        TypedArray::I32(x) => ("i32", x.iter().map(i32::to_string).collect()),
        TypedArray::I64(x) => ("i64", x.iter().map(i64::to_string).collect()),
        TypedArray::U8(x) => ("u8", x.iter().map(u8::to_string).collect()),
        TypedArray::U16(x) => ("u16", x.iter().map(u16::to_string).collect()),
        TypedArray::U32(x) => ("u32", x.iter().map(u32::to_string).collect()),
        TypedArray::U64(x) => ("u64", x.iter().map(u64::to_string).collect()),
        TypedArray::F32(x) => ("f32", x.iter().map(|v| py_float(f64::from(*v))).collect()),
        TypedArray::F64(x) => ("f64", x.iter().map(|v| py_float(*v)).collect()),
        TypedArray::Str(x) => ("str", x.iter().map(|s| py_json(s)).collect()),
        other => panic!("unexpected {other:?}"),
    };
    format!("{dtype} {shape} {}", vals.join(" "))
        .trim_end()
        .to_string()
}

fn walk(g: &Group, lines: &mut Vec<String>) {
    lines.push(format!("G {}", g.path()));
    let at = |p: &str| {
        if p == "/" {
            "/".to_string()
        } else {
            p.to_string()
        }
    };
    for a in g.attrs().unwrap() {
        lines.push(format!(
            "A {}@{} {}",
            at(g.path()),
            a.name,
            dump_array(&a.value().unwrap(), a.shape.as_deref() == Some(&[]))
        ));
    }
    for (_, node) in g.members().unwrap() {
        match node {
            Node::Group(child) => walk(&child, lines),
            Node::Dataset(d) => {
                let v = d.read().unwrap_or_else(|e| panic!("{}: {e}", d.path()));
                lines.push(format!(
                    "D {} {}",
                    d.path(),
                    dump_array(&v, d.shape().is_empty())
                ));
                for a in d.attrs().unwrap() {
                    lines.push(format!(
                        "A {}@{} {}",
                        d.path(),
                        a.name,
                        dump_array(&a.value().unwrap(), a.shape.as_deref() == Some(&[]))
                    ));
                }
            }
            _ => {}
        }
    }
}

fn check(name: &str) {
    let file = File::open(fixtures().join(format!("{name}.h5"))).unwrap();
    let mut lines = Vec::new();
    walk(&file.root().unwrap(), &mut lines);
    lines.sort();
    let expected = std::fs::read_to_string(fixtures().join(format!("{name}.dump"))).unwrap();
    let expected: Vec<&str> = expected.lines().collect();
    for (got, want) in lines.iter().zip(&expected) {
        assert_eq!(got, want, "{name}");
    }
    assert_eq!(lines.len(), expected.len(), "{name}: line count");
}

#[test]
fn earliest() {
    check("earliest");
}

#[test]
fn earliest_big_endian() {
    check("earliest_be");
}

#[test]
fn v18() {
    check("v18");
}

#[test]
fn v18_big_endian() {
    check("v18_be");
}

#[test]
fn latest() {
    check("latest");
}

#[test]
fn latest_big_endian() {
    check("latest_be");
}

#[test]
fn creation_ordered_dense_group() {
    check("track_order");
    // Members come back in creation order, not name order.
    let names: Vec<String> = File::open(fixtures().join("track_order.h5"))
        .unwrap()
        .root()
        .unwrap()
        .members()
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names.first().map(String::as_str), Some("g00"));
    assert_eq!(names.len(), 20);
}

#[test]
fn python_formatting_helpers() {
    assert_eq!(py_float(1.0), "1.0");
    assert_eq!(py_float(0.1), "0.1");
    assert_eq!(py_float(1e-5), "1e-05");
    assert_eq!(py_float(1.5e20), "1.5e+20");
    assert_eq!(py_float(f64::NAN), "nan");
    assert_eq!(py_json("αβγ"), "\"\\u03b1\\u03b2\\u03b3\"");
}

fn walk_all(g: &Group, depth: usize) {
    if depth > 8 {
        return;
    }
    let _ = g
        .attrs()
        .map(|a| a.iter().filter(|x| x.value().is_ok()).count());
    if let Ok(members) = g.members() {
        for (_, n) in members {
            match n {
                Node::Group(c) => walk_all(&c, depth + 1),
                Node::Dataset(d) => {
                    let _ = d.read();
                    let _ = d.attrs();
                }
                _ => {}
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Corrupting any bytes of a real file never panics the reader.
    #[test]
    fn corrupted_files_never_panic(which in 0usize..3, edits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..16)) {
        let name = ["earliest.h5", "v18.h5", "latest.h5"][which];
        let mut bytes = std::fs::read(fixtures().join(name)).unwrap();
        let len = bytes.len();
        for (pos, b) in edits {
            bytes[pos % len] = b;
        }
        if let Ok(f) = File::from_bytes(bytes).map(|f| f.with_load_limit(1 << 22)) {
            if let Ok(root) = f.root() {
                walk_all(&root, 0);
            }
        }
    }
}

#[test]
fn latest_uses_every_chunk_index() {
    let f = File::open(fixtures().join("latest.h5")).unwrap();
    let mut kinds = Vec::new();
    for (name, n) in f.root().unwrap().members().unwrap() {
        if let Node::Dataset(d) = n {
            let dbg = format!("{d:?}");
            for k in [
                "BTreeV1",
                "Single",
                "Implicit",
                "FixedArray",
                "ExtensibleArray",
                "BTreeV2",
            ] {
                if dbg.contains(&format!("index: {k}")) {
                    kinds.push((name.clone(), k));
                }
            }
        }
    }
    eprintln!("{kinds:?}");
    for k in [
        "Single",
        "Implicit",
        "FixedArray",
        "ExtensibleArray",
        "BTreeV2",
    ] {
        assert!(
            kinds.iter().any(|(_, x)| *x == k),
            "no {k} index in latest.h5: {kinds:?}"
        );
    }
}

/// Our lookup3 reproduces the checksums the HDF5 library stored in real files.
#[test]
fn checksums_match_hdf5() {
    for name in ["v18.h5", "latest.h5"] {
        let bytes = std::fs::read(fixtures().join(name)).unwrap();
        // Superblock v2/v3: 44 bytes (with 8-byte offsets) then the checksum.
        let stored = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
        assert_eq!(
            xr_hdf5::lookup3(&bytes[..44], 0),
            stored,
            "{name} superblock"
        );
        // Every v2 object header chunk 0: "OHDR" … checksum.
        let mut checked = 0;
        for i in 0..bytes.len().saturating_sub(4) {
            if &bytes[i..i + 4] == b"OHDR" && bytes[i + 4] == 2 {
                let flags = bytes[i + 5];
                let mut p = i + 6;
                if flags & 0x20 != 0 {
                    p += 16;
                }
                if flags & 0x10 != 0 {
                    p += 4;
                }
                let w = 1usize << (flags & 3);
                let mut size = 0usize;
                for (j, b) in bytes[p..p + w].iter().enumerate() {
                    size |= (*b as usize) << (8 * j);
                }
                let end = p + w + size;
                if end + 4 > bytes.len() {
                    continue;
                }
                let stored = u32::from_le_bytes(bytes[end..end + 4].try_into().unwrap());
                if xr_hdf5::lookup3(&bytes[i..end], 0) == stored {
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 10,
            "{name}: only {checked} object headers verified"
        );
    }
}
