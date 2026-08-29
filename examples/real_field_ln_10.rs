use linalg_traits::RealField;

fn ln_10<T: RealField>() -> T {
    T::ln_10()
}

fn main() {
    assert_eq!(ln_10::<f64>(), std::f64::consts::LN_10);
}
