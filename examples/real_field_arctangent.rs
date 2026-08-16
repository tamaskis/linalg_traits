use linalg_traits::RealField;

fn atan<T: RealField>(x: T) -> T {
    let y1: T = x.atan();
    let y2: T = T::atan(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(atan(0_f64), 0_f64);
}
