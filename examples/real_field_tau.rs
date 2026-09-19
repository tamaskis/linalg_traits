use linalg_traits::RealField;

fn tau<T: RealField>() -> T {
    T::tau()
}

fn main() {
    assert_eq!(tau::<f64>(), std::f64::consts::TAU);
}
