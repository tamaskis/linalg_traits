use linalg_traits::RealField;

fn powf<T: RealField>(x: T, n: T) -> T {
    let y1: T = x.powf(n);
    let y2: T = T::powf(x, n);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(powf(2_f64, 3_f64), 8_f64);
}
