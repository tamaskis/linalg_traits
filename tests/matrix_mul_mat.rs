use linalg_traits::{Matrix, MulMat};

#[test]
fn test_mat() {
    let a = linalg_traits::Mat::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = linalg_traits::Mat::<f64>::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = a.mul_mat(&b);
    assert_eq!(c.as_slice().as_ref(), &[58.0, 64.0, 139.0, 154.0]);
}

#[cfg(feature = "nalgebra")]
#[test]
fn test_nalgebra() {
    use nalgebra::{DMatrix, SMatrix};
    let s = SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sr = SMatrix::<f64, 3, 2>::from_row_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let d = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let dr = DMatrix::<f64>::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let expected = DMatrix::<f64>::from_row_slice(2, 2, &[58.0, 64.0, 139.0, 154.0]);

    assert_eq!(
        s.mul_mat(&sr),
        SMatrix::<f64, 2, 2>::from_row_slice(&[58.0, 64.0, 139.0, 154.0])
    );
    assert_eq!(s.mul_mat(&dr), expected);
    assert_eq!(d.mul_mat(&sr), expected);
    assert_eq!(d.mul_mat(&dr), expected);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_ndarray() {
    use ndarray::array;
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    assert_eq!(a.mul_mat(&b), array![[58.0, 64.0], [139.0, 154.0]]);
}

#[cfg(feature = "faer")]
#[test]
fn test_faer() {
    use faer::mat;
    let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = mat![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    assert_eq!(a.mul_mat(&b), mat![[58.0, 64.0], [139.0, 154.0]]);
}
