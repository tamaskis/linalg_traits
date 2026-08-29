use linalg_traits::RealField;

fn acosd<T: RealField>(x: T) -> T {
    let y1: T = x.acosd();
    let y2: T = T::acosd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acosd(1_f64), 0_f64);
}
