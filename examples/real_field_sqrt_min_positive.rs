use linalg_traits::RealField;

fn sqrt_min_positive<T: RealField>() -> T {
    T::sqrt_min_positive()
}

fn main() {
    assert_eq!(sqrt_min_positive::<f64>(), f64::MIN_POSITIVE.sqrt());
}
