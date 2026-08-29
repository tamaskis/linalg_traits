use linalg_traits::RealField;

fn frac_2_sqrt_pi<T: RealField>() -> T {
    T::frac_2_sqrt_pi()
}

fn main() {
    assert_eq!(frac_2_sqrt_pi::<f64>(), std::f64::consts::FRAC_2_SQRT_PI);
}
