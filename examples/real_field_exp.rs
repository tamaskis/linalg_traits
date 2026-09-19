use linalg_traits::RealField;

fn exp<T: RealField>(x: T) -> T {
    let y1: T = x.exp();
    let y2: T = T::exp(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(exp(0_f64), 1_f64);
}
