use linalg_traits::RealField;

fn cosd<T: RealField>(x: T) -> T {
    let y1: T = x.cosd();
    let y2: T = T::cosd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cosd(0_f64), 1_f64);
}
