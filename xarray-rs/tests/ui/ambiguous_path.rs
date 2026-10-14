use xarray_rs::{Variable, isel};

fn f(v: &Variable) {
    let _ = isel!(v[a::B]);
}

fn main() {}
