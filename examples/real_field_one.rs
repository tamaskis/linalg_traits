use linalg_traits::RealField;

fn one<T: RealField>() -> T {
    T::one()
}

fn main() {
    assert_eq!(one::<f64>(), 1_f64);
}
