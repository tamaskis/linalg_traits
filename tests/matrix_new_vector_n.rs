#[cfg(feature = "faer")]
use faer::{Col, Mat as FMat};
#[cfg(feature = "faer")]
use linalg_traits::Vector;
use linalg_traits::{Mat, Matrix};
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};
use numtest::*;

const M: usize = 3;
const N: usize = 2;

#[test]
fn test_vec() {
    let mat: Mat<f64> = Mat::new_with_shape(M, N);
    let vec: Vec<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec, [0.0; N]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mat: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let vec: DVector<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec, [0.0; N]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mat: SMatrix<f64, M, N> = SMatrix::new_with_shape(M, N);
    let vec: SVector<f64, N> = mat.new_vector_n();
    assert_arrays_equal!(vec, [0.0; N]);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mat: Array2<f64> = Array2::new_with_shape(M, N);
    let vec: Array1<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec, [0.0; N]);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_col() {
    let mat: FMat<f64> = FMat::new_with_shape(M, N);
    let vec: Col<f64> = mat.new_vector_n();
    assert_arrays_equal!(Vector::as_slice(&vec), [0.0; N]);
}
