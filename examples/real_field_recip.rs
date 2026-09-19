use linalg_traits::RealField;

fn recip<T: RealField>(x: T) -> T {
    let y1: T = x.recip();
    let y2: T = T::recip(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(recip(4_f64), 0.25_f64);
}
