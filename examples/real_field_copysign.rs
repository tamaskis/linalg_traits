use linalg_traits::RealField;

fn copysign<T: RealField>(x: T, sign: T) -> T {
    let y1: T = x.copysign(sign);
    let y2: T = T::copysign(x, sign);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(copysign(3_f64, -1_f64), -3_f64);
}
