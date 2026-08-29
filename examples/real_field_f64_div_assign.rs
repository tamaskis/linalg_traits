use linalg_traits::RealField;

fn div_assign<T: RealField>(x: T, y: f64) -> T {
    let mut z1: T = x;
    z1 /= y;

    let mut z2: T = x;
    z2 /= &y;

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(div_assign(6_f64, 2_f64), 3_f64);
}
