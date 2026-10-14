use xarray_rs::{Variable, isel};

fn f(v: &Variable) {
    let _ = isel!(v[1:2:3:4]);
}

fn main() {}
