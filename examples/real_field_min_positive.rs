use linalg_traits::RealField;

fn min_positive<T: RealField>() -> T {
    T::min_positive()
}

fn main() {
    assert_eq!(min_positive::<f64>(), f64::MIN_POSITIVE);
}
