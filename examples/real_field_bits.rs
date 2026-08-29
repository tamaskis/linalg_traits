use linalg_traits::RealField;

fn bits<T: RealField>() -> usize {
    T::bits()
}

fn main() {
    assert_eq!(bits::<f64>(), 64);
}
