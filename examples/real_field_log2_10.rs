use linalg_traits::RealField;

fn log2_10<T: RealField>() -> T {
    T::log2_10()
}

fn main() {
    assert_eq!(log2_10::<f64>(), std::f64::consts::LOG2_10);
}
