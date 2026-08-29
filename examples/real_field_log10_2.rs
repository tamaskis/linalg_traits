use linalg_traits::RealField;

fn log10_2<T: RealField>() -> T {
    T::log10_2()
}

fn main() {
    assert_eq!(log10_2::<f64>(), std::f64::consts::LOG10_2);
}
