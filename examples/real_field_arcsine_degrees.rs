use linalg_traits::RealField;

fn asind<T: RealField>(x: T) -> T {
    let y1: T = x.asind();
    let y2: T = T::asind(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asind(0_f64), 0_f64);
}
