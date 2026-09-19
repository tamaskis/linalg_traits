use linalg_traits::RealField;

fn log2_e<T: RealField>() -> T {
    T::log2_e()
}

fn main() {
    assert_eq!(log2_e::<f64>(), std::f64::consts::LOG2_E);
}
