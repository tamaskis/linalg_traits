use linalg_traits::RealField;

fn as_slice<T: RealField>(x: &T) -> &[f64] {
    let y1: &[f64] = x.as_slice();
    let y2: &[f64] = T::as_slice(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    let x = 2_f64;
    assert_eq!(as_slice(&x), &[2_f64]);
}
