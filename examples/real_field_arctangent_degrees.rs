use linalg_traits::RealField;

fn atand<T: RealField>(x: T) -> T {
    let y1: T = x.atand();
    let y2: T = T::atand(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(atand(0_f64), 0_f64);
}
