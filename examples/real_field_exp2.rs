use linalg_traits::RealField;

fn exp2<T: RealField>(x: T) -> T {
    let y1: T = x.exp2();
    let y2: T = T::exp2(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(exp2(3_f64), 8_f64);
}
