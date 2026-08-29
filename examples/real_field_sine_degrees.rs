use linalg_traits::RealField;

fn sind<T: RealField>(x: T) -> T {
    let y1: T = x.sind();
    let y2: T = T::sind(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sind(0_f64), 0_f64);
}
