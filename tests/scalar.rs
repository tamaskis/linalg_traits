use linalg_traits::Scalar;

#[test]
fn test_new() {
    let x: f64 = <f64 as Scalar>::new(5.0);
    assert_eq!(x, 5.0);
}
