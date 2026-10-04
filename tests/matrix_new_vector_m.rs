#[cfg(feature = "faer")]
use faer::{Mat as FMat, Row};
use linalg_traits::{Mat, Matrix, RowVector};
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, RowDVector, RowSVector, SMatrix};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};
use numtest::*;

const M: usize = 3;
const N: usize = 2;

#[test]
fn test_vec() {
    let mat: Mat<f64> = Mat::new_with_shape(M, N);
    let vec: Vec<f64> = mat.new_vector_m();
    assert_arrays_equal!(vec, [0.0; M]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mat: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let vec: RowDVector<f64> = mat.new_vector_m();
    assert_arrays_equal!(vec, [0.0; M]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mat: SMatrix<f64, M, N> = SMatrix::new_with_shape(M, N);
    let vec: RowSVector<f64, M> = mat.new_vector_m();
    assert_arrays_equal!(vec, [0.0; M]);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mat: Array2<f64> = Array2::new_with_shape(M, N);
    let vec: Array1<f64> = mat.new_vector_m();
    assert_arrays_equal!(vec, [0.0; M]);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_row() {
    let mat: FMat<f64> = FMat::new_with_shape(M, N);
    let vec: Row<f64> = mat.new_vector_m();
    assert_eq!(RowVector::len(&vec), M);
    assert_eq!(vec[0], 0.0);
}
