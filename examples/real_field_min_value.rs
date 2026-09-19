use linalg_traits::RealField;

fn min_value<T: RealField>() -> Option<T> {
    T::min_value()
}

fn main() {
    assert_eq!(min_value::<f64>(), Some(f64::MIN));
}
