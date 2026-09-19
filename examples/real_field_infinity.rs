use linalg_traits::RealField;

fn infinity<T: RealField>() -> T {
    T::infinity()
}

fn main() {
    assert_eq!(infinity::<f64>(), f64::INFINITY);
}
