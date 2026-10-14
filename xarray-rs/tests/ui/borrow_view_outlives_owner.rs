// A view cannot outlive the variable it borrows from.
use xarray_rs::ndarray::{ArrayD, IxDyn};
use xarray_rs::{Variable, VariableView};

fn make() -> VariableView<'static> {
    let v = Variable::new(["x"], ArrayD::<f64>::zeros(IxDyn(&[3]))).unwrap();
    v.view()
}

fn main() {}
