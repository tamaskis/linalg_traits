use linalg_traits::RealField;

fn e<T: RealField>() -> T {
    T::e()
}

fn main() {
    assert_eq!(e::<f64>(), std::f64::consts::E);
}
