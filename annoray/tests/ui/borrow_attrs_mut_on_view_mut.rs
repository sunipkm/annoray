// Mutable views can change element data only; attributes stay read-only.
use annoray::VariableViewMut;

fn f(mut v: VariableViewMut<'_>) {
    let _ = v.attrs_mut();
}

fn main() {}
