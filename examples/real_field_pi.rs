use linalg_traits::RealField;

fn pi<T: RealField>() -> T {
    T::pi()
}

fn main() {
    assert_eq!(pi::<f64>(), std::f64::consts::PI);
}
