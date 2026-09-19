use linalg_traits::RealField;

#[allow(clippy::op_ref)]
fn sub_rhs<T: RealField>(x: T, y: f64) -> T {
    let z1: T = x - y;
    let z2: T = x - &y;
    let z3: T = &x - y;
    let z4: T = &x - &y;

    assert_eq!(z1, z2);
    assert_eq!(z2, z3);
    assert_eq!(z3, z4);

    z1
}

#[allow(clippy::op_ref)]
fn sub_lhs<T: RealField>(x: f64, y: T) -> T {
    let z1: T = x - y;
    let z2: T = x - &y;
    let z3: T = &x - y;
    let z4: T = &x - &y;

    assert_eq!(z1, z2);
    assert_eq!(z2, z3);
    assert_eq!(z3, z4);

    z1
}

fn main() {
    assert_eq!(sub_rhs(3_f64, 2_f64), 1_f64);
    assert_eq!(sub_lhs(3_f64, 2_f64), 1_f64);
}
