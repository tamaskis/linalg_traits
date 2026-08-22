use linalg_traits::RealField;

#[test]
fn test_new() {
    let x: f64 = <f64 as RealField>::new(5.0);
    assert_eq!(x, 5.0);
}
