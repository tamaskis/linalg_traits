use linalg_traits::RealField;

fn frac_pi_2<T: RealField>() -> T {
    T::frac_pi_2()
}

fn main() {
    assert_eq!(frac_pi_2::<f64>(), std::f64::consts::FRAC_PI_2);
}
