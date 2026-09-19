use linalg_traits::RealField;

fn sqrt_max_positive<T: RealField>() -> T {
    T::sqrt_max_positive()
}

fn main() {
    assert_eq!(sqrt_max_positive::<f64>(), f64::MAX.sqrt());
}
