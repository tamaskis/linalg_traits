use linalg_traits::RealField;

fn log10_e<T: RealField>() -> T {
    T::log10_e()
}

fn main() {
    assert_eq!(log10_e::<f64>(), std::f64::consts::LOG10_E);
}
