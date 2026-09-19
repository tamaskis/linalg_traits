use linalg_traits::RealField;

fn two_pi<T: RealField>() -> T {
    T::two_pi()
}

fn main() {
    assert_eq!(two_pi::<f64>(), std::f64::consts::TAU);
}
