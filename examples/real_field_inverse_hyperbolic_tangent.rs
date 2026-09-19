use linalg_traits::RealField;

fn atanh<T: RealField>(x: T) -> T {
    let y1: T = x.atanh();
    let y2: T = T::atanh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(atanh(0_f64), 0_f64);
}
