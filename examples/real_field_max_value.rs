use linalg_traits::RealField;

fn max_value<T: RealField>() -> Option<T> {
    T::max_value()
}

fn main() {
    assert_eq!(max_value::<f64>(), Some(f64::MAX));
}
