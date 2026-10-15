// Two live mutable views of the same variable must be rejected.
use annoray::Variable;

fn f(v: &mut Variable) {
    let a = v.view_mut();
    let b = v.view_mut();
    drop((a, b));
}

fn main() {}
