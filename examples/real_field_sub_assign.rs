use linalg_traits::RealField;

fn sub_assign<T: RealField>(x: T, y: T) -> T {
    let mut z1: T = x;
    z1 -= y;

    let mut z2: T = x;
    z2 -= &y;

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(sub_assign(3_f64, 2_f64), 1_f64);
}
