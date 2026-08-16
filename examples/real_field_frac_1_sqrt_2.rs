use linalg_traits::RealField;

fn frac_1_sqrt_2<T: RealField>() -> T {
    T::frac_1_sqrt_2()
}

fn main() {
    assert_eq!(frac_1_sqrt_2::<f64>(), std::f64::consts::FRAC_1_SQRT_2);
}
