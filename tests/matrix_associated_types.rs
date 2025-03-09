use linalg_traits::{Mat, Matrix};
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Dimensions for all unit tests.
const M: usize = 3;
const N: usize = 2;

#[test]
fn test_vec_from_mat() {
    let mat: Mat<f64> = <Mat<f64> as Matrix<f64>>::new_with_shape(M, N);
    let vec_m: Vec<f64> = mat.new_vector_m();
    let vec_n: Vec<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec_m, [0.0; M]);
    assert_arrays_equal!(vec_n, [0.0; N]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_from_dmatrix() {
    let mat: DMatrix<f64> = <DMatrix<f64> as Matrix<f64>>::new_with_shape(M, N);
    let vec_m: DVector<f64> = mat.new_vector_m();
    let vec_n: DVector<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec_m, [0.0; M]);
    assert_arrays_equal!(vec_n, [0.0; N]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector_from_smatrix() {
    let mat: SMatrix<f64, M, N> = <SMatrix<f64, M, N> as Matrix<f64>>::new_with_shape(M, N);
    let vec_m: SVector<f64, M> = mat.new_vector_m();
    let vec_n: SVector<f64, N> = mat.new_vector_n();
    assert_arrays_equal!(vec_m, [0.0; M]);
    assert_arrays_equal!(vec_n, [0.0; N]);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_from_array2() {
    let mat: Array2<f64> = <Array2<f64> as Matrix<f64>>::new_with_shape(M, N);
    let vec_m: Array1<f64> = mat.new_vector_m();
    let vec_n: Array1<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec_m, [0.0; M]);
    assert_arrays_equal!(vec_n, [0.0; N]);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat_from_faer_mat() {
    let mat: FMat<f64> = FMat::new_with_shape(M, N);
    let vec_m: FMat<f64> = mat.new_vector_m();
    let vec_n: FMat<f64> = mat.new_vector_n();
    assert_arrays_equal!(vec_m.as_slice(), [0.0; M]);
    assert_arrays_equal!(vec_n.as_slice(), [0.0; N]);
}
