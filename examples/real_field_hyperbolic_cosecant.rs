use linalg_traits::RealField;

fn csch<T: RealField>(x: T) -> T {
    let y1: T = x.csch();
    let y2: T = T::csch(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(csch(1_f64), 1_f64.sinh().recip());
}
