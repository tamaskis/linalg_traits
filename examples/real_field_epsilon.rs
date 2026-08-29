use linalg_traits::RealField;

fn epsilon<T: RealField>() -> T {
    T::epsilon()
}

fn main() {
    assert_eq!(epsilon::<f64>(), f64::EPSILON);
}
