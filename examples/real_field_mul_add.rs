use linalg_traits::RealField;

fn mul_add<T: RealField>(x: T, a: T, b: T) -> T {
    let y1: T = x.mul_add(a, b);
    let y2: T = T::mul_add(x, a, b);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(mul_add(2_f64, 3_f64, 4_f64), 10_f64);
}
