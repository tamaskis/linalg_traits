use linalg_traits::RealField;

fn ln_2<T: RealField>() -> T {
    T::ln_2()
}

fn main() {
    assert_eq!(ln_2::<f64>(), std::f64::consts::LN_2);
}
