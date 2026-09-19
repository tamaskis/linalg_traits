use linalg_traits::RealField;

fn nan<T: RealField>() -> T {
    T::nan()
}

fn main() {
    assert!(nan::<f64>().is_nan());
}
