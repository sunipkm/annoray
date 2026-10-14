use xarray_rs::{Variable, isel};

fn f(v: &Variable) {
    let _ = isel!(v[time = 0, 1]);
}

fn main() {}
