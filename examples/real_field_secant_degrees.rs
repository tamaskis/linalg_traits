use linalg_traits::RealField;

fn secd<T: RealField>(x: T) -> T {
    let y1: T = x.secd();
    let y2: T = T::secd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(secd(30_f64), 30_f64.to_radians().sec());
}
