use linalg_traits::{Matrix, RowVectorMulMat};

#[test]
fn test_vec() {
    let a = linalg_traits::Mat::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(vec![1.0, 2.0].mul_mat(&a), vec![9.0, 12.0, 15.0]);
}

#[test]
#[should_panic(expected = "incompatible shapes")]
fn test_vec_incompatible() {
    let a = linalg_traits::Mat::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let _ = vec![1.0, 2.0, 3.0].mul_mat(&a);
}

#[cfg(feature = "nalgebra")]
#[test]
fn test_nalgebra() {
    use nalgebra::{DMatrix, RowDVector, RowSVector, SMatrix};
    let s = SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let d = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sv = RowSVector::<f64, 2>::from_row_slice(&[1.0, 2.0]);
    let dv = RowDVector::<f64>::from_row_slice(&[1.0, 2.0]);
    let expected = RowDVector::<f64>::from_row_slice(&[9.0, 12.0, 15.0]);

    assert_eq!(
        sv.mul_mat(&s),
        RowSVector::<f64, 3>::from_row_slice(&[9.0, 12.0, 15.0])
    );
    assert_eq!(sv.mul_mat(&d), expected);
    assert_eq!(dv.mul_mat(&s), expected);
    assert_eq!(dv.mul_mat(&d), expected);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_ndarray() {
    use ndarray::array;
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(array![1.0, 2.0].mul_mat(&a), array![9.0, 12.0, 15.0]);
}

#[cfg(feature = "faer")]
#[test]
fn test_faer() {
    use faer::{Row, mat};
    let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let v = Row::<f64>::from_fn(2, |i| (i + 1) as f64);
    assert_eq!(
        v.mul_mat(&a),
        Row::<f64>::from_fn(3, |i| [9.0, 12.0, 15.0][i])
    );
}
