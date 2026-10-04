use linalg_traits::{Matrix, MulVec};

#[test]
fn test_mat() {
    let a = linalg_traits::Mat::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(a.mul_vec(&vec![1.0, 2.0, 3.0]), vec![14.0, 32.0]);
}

#[cfg(feature = "nalgebra")]
#[test]
fn test_nalgebra() {
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};
    let s = SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let d = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sv = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
    let dv = DVector::<f64>::from_row_slice(&[1.0, 2.0, 3.0]);
    let expected = DVector::<f64>::from_row_slice(&[14.0, 32.0]);

    assert_eq!(
        s.mul_vec(&sv),
        SVector::<f64, 2>::from_row_slice(&[14.0, 32.0])
    );
    assert_eq!(s.mul_vec(&dv), expected);
    assert_eq!(d.mul_vec(&sv), expected);
    assert_eq!(d.mul_vec(&dv), expected);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_ndarray() {
    use ndarray::array;
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(a.mul_vec(&array![1.0, 2.0, 3.0]), array![14.0, 32.0]);
}

#[cfg(feature = "faer")]
#[test]
fn test_faer() {
    use faer::{Col, mat};
    let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let v = Col::<f64>::from_fn(3, |i| (i + 1) as f64);
    assert_eq!(a.mul_vec(&v), Col::<f64>::from_fn(2, |i| [14.0, 32.0][i]));
}
