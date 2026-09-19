use linalg_traits::RealField;

fn tan<T: RealField>(x: T) -> T {
    let y1: T = x.tan();
    let y2: T = T::tan(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(tan(0_f64), 0_f64);
}
