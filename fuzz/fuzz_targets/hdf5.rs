//! The HDF5 reader on its own: every dataset of every group read whole, under a load limit.
#![no_main]

use annoray::hdf5::{File, Group, Node};
use libfuzzer_sys::fuzz_target;

fn walk(g: &Group, depth: usize) {
    if depth > 8 {
        return;
    }
    let Ok(members) = g.members() else { return };
    for (_, node) in members {
        match node {
            Node::Dataset(d) => {
                let _ = d.attrs();
                let _ = d.read();
            }
            Node::Group(c) => walk(&c, depth + 1),
            _ => {}
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if let Ok(f) = File::from_bytes(data.to_vec()) {
        let f = f.with_load_limit(16 << 20);
        if let Ok(root) = f.root() {
            let _ = root.attrs();
            walk(&root, 0);
        }
    }
});
