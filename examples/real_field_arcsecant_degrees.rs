use linalg_traits::RealField;

fn asecd<T: RealField>(x: T) -> T {
    let y1: T = x.asecd();
    let y2: T = T::asecd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asecd(2_f64), 2_f64.asec().to_degrees());
}
