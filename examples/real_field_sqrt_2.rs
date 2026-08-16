use linalg_traits::RealField;

fn sqrt_2<T: RealField>() -> T {
    T::sqrt_2()
}

fn main() {
    assert_eq!(sqrt_2::<f64>(), std::f64::consts::SQRT_2);
}
