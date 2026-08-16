use linalg_traits::RealField;

fn sind_cosd<T: RealField>(x: T) -> (T, T) {
    let y1: (T, T) = x.sind_cosd();
    let y2: (T, T) = T::sind_cosd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sind_cosd(0_f64), (0_f64, 1_f64));
}
