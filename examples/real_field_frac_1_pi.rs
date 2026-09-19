use linalg_traits::RealField;

fn frac_1_pi<T: RealField>() -> T {
    T::frac_1_pi()
}

fn main() {
    assert_eq!(frac_1_pi::<f64>(), std::f64::consts::FRAC_1_PI);
}
