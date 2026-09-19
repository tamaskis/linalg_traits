use linalg_traits::RealField;

fn cscd<T: RealField>(x: T) -> T {
    let y1: T = x.cscd();
    let y2: T = T::cscd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cscd(30_f64), 30_f64.to_radians().csc());
}
