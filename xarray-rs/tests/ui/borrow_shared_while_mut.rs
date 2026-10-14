// A shared view blocks mutable selection until it is dropped.
use xarray_rs::{Variable, isel};

fn f(v: &mut Variable) {
    let shared = isel!(v[0]).unwrap();
    let _m = isel!(mut v[1]).unwrap();
    drop(shared);
}

fn main() {}
