use linalg_traits::RealField;

fn acos<T: RealField>(x: T) -> T {
    let y1: T = x.acos();
    let y2: T = T::acos(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acos(1_f64), 0_f64);
}
