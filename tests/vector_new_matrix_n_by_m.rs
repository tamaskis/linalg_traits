#[cfg(feature = "faer")]
use faer::{Col, Mat as FMat};
use linalg_traits::{Mat, Matrix, Vector};
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};
use numtest::*;

const M: usize = 3;
const N: usize = 2;

#[test]
fn test_vec() {
    let vec: Vec<f64> = Vec::new_with_length(N);
    let mat: Mat<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let expected: Mat<f64> = Mat::new_with_shape(N, M);
    assert_arrays_equal!(mat, expected);
    assert_eq!(mat.shape(), (N, M));
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    let vec: DVector<f64> = DVector::new_with_length(N);
    let mat: DMatrix<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let expected: DMatrix<f64> = DMatrix::new_with_shape(N, M);
    assert_arrays_equal!(mat, expected);
    assert_eq!(mat.shape(), (N, M));
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    let vec: SVector<f64, N> = SVector::new_with_length(N);
    let mat: SMatrix<f64, N, M> = vec.new_matrix_n_by_m::<M>(None);
    let expected: SMatrix<f64, N, M> = SMatrix::new_with_shape(N, M);
    assert_arrays_equal!(mat, expected);
    assert_eq!(mat.shape(), (N, M));
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray() {
    let vec: Array1<f64> = Array1::new_with_length(N);
    let mat: Array2<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let expected: Array2<f64> = Array2::new_with_shape(N, M);
    assert_arrays_equal!(mat, expected);
    assert_eq!(Matrix::shape(&mat), (N, M));
}

#[test]
#[cfg(feature = "faer")]
fn test_faer() {
    let vec: Col<f64> = Col::new_with_length(N);
    let mat: FMat<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let expected: FMat<f64> = FMat::new_with_shape(N, M);
    assert_arrays_equal!(Matrix::as_slice(&mat), Matrix::as_slice(&expected));
    assert_eq!(Matrix::shape(&mat), (N, M));
}
