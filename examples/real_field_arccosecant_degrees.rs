use linalg_traits::RealField;

fn acscd<T: RealField>(x: T) -> T {
    let y1: T = x.acscd();
    let y2: T = T::acscd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acscd(2_f64), 2_f64.acsc().to_degrees());
}
