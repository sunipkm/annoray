//! Print the structure of an HDF5 file: `cargo run -p xr-hdf5 --example inspect -- FILE`.
#![allow(clippy::print_stdout)]

use xr_hdf5::{Attribute, File, Group, Node};

fn show_attr(indent: &str, a: &Attribute) {
    let value = match (a.references(), a.reference_lists(), a.value()) {
        (Ok(r), _, _) => format!("refs {r:?}"),
        (_, Ok(l), _) => format!("ref lists {l:?}"),
        (_, _, Ok(v)) => format!("{v:?}").chars().take(100).collect(),
        (_, _, Err(e)) => format!("<{e}>"),
    };
    println!(
        "{indent}@{} {:?} shape={:?}: {value}",
        a.name, a.dtype, a.shape
    );
}

fn walk(g: &Group, indent: &str) -> Result<(), xr_io::IoError> {
    println!("{indent}{} (group @{})", g.path(), g.address());
    for a in g.attrs()? {
        show_attr(&format!("{indent}  "), &a);
    }
    for (name, node) in g.members()? {
        match node {
            Node::Group(c) => walk(&c, &format!("{indent}  "))?,
            Node::Dataset(d) => {
                println!(
                    "{indent}  {name} @{} {:?} shape={:?} max={:?} chunks={:?} fill={:?}",
                    d.address(),
                    d.dtype(),
                    d.shape(),
                    d.max_shape(),
                    d.chunks(),
                    d.fill_value()
                );
                for a in d.attrs()? {
                    show_attr(&format!("{indent}    "), &a);
                }
            }
            _ => println!("{indent}  {name} (other)"),
        }
    }
    Ok(())
}

fn main() -> Result<(), xr_io::IoError> {
    for path in std::env::args().skip(1) {
        let f = File::open(&path)?;
        println!("== {path} (superblock v{})", f.superblock_version());
        walk(&f.root()?, "")?;
    }
    Ok(())
}
