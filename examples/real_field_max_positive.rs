use linalg_traits::RealField;

fn max_positive<T: RealField>() -> T {
    T::max_positive()
}

fn main() {
    assert_eq!(max_positive::<f64>(), f64::MAX);
}
