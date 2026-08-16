use linalg_traits::RealField;

fn tand<T: RealField>(x: T) -> T {
    let y1: T = x.tand();
    let y2: T = T::tand(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(tand(0_f64), 0_f64);
}
