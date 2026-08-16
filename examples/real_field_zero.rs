use linalg_traits::RealField;

fn zero<T: RealField>() -> T {
    T::zero()
}

fn main() {
    assert_eq!(zero::<f64>(), 0_f64);
}
