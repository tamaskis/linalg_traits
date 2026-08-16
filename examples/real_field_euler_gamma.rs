use linalg_traits::RealField;

fn euler_gamma<T: RealField>() -> T {
    T::euler_gamma()
}

fn main() {
    assert_eq!(euler_gamma::<f64>(), 0.577_215_664_901_532_9);
}
