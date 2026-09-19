use linalg_traits::RealField;

fn cotd<T: RealField>(x: T) -> T {
    let y1: T = x.cotd();
    let y2: T = T::cotd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cotd(30_f64), 30_f64.to_radians().cot());
}
