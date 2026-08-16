use linalg_traits::RealField;

fn frac_pi_8<T: RealField>() -> T {
    T::frac_pi_8()
}

fn main() {
    assert_eq!(frac_pi_8::<f64>(), std::f64::consts::FRAC_PI_8);
}
