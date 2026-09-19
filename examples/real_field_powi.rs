use linalg_traits::RealField;

fn powi<T: RealField>(x: T, n: i32) -> T {
    let y1: T = x.powi(n);
    let y2: T = T::powi(x, n);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(powi(2_f64, 3), 8_f64);
}
