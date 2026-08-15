#[cfg(feature = "faer")]
use faer::Mat as FMat;
use linalg_traits::Matrix;
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};
#[cfg(feature = "ndarray")]
use ndarray::Array2;

// Test conditions.
static X: &[f64; 6] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn test_mat() {
    let x = <linalg_traits::Mat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_eq!(x[(0, 0)], 1.0);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(0, 2)], 3.0);
    assert_eq!(x[(1, 0)], 4.0);
    assert_eq!(x[(1, 1)], 5.0);
    assert_eq!(x[(1, 2)], 6.0);
}

#[test]
#[should_panic]
fn test_mat_out_of_bounds() {
    let x = <linalg_traits::Mat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    _ = x[(2, 0)];
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    let x = <DMatrix<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_eq!(x[(0, 0)], 1.0);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(0, 2)], 3.0);
    assert_eq!(x[(1, 0)], 4.0);
    assert_eq!(x[(1, 1)], 5.0);
    assert_eq!(x[(1, 2)], 6.0);
}

#[test]
#[should_panic]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix_out_of_bounds() {
    let x = <DMatrix<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    _ = x[(0, 3)];
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    let x = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_eq!(x[(0, 0)], 1.0);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(0, 2)], 3.0);
    assert_eq!(x[(1, 0)], 4.0);
    assert_eq!(x[(1, 1)], 5.0);
    assert_eq!(x[(1, 2)], 6.0);
}

#[test]
#[should_panic]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_out_of_bounds() {
    let x = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(2, 3, X);
    _ = x[(2, 0)];
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    let x = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_eq!(x[(0, 0)], 1.0);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(0, 2)], 3.0);
    assert_eq!(x[(1, 0)], 4.0);
    assert_eq!(x[(1, 1)], 5.0);
    assert_eq!(x[(1, 2)], 6.0);
}

#[test]
#[should_panic]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2_out_of_bounds() {
    let x = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    _ = x[(0, 3)];
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let x = <FMat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_eq!(x[(0, 0)], 1.0);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(0, 2)], 3.0);
    assert_eq!(x[(1, 0)], 4.0);
    assert_eq!(x[(1, 1)], 5.0);
    assert_eq!(x[(1, 2)], 6.0);
}

#[test]
#[should_panic]
#[cfg(feature = "faer")]
fn test_faer_mat_out_of_bounds() {
    let x = <FMat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    _ = x[(2, 0)];
}
