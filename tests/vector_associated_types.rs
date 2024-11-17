use linalg_traits::{Mat, Matrix, Vector};
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::{Array1, Array2};

// Dimensions for all unit tests.
const M: usize = 3;
const N: usize = 2;

#[test]
fn test_mat_from_vec() {
    // Vector.
    let vec: Vec<f64> = Vec::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: Mat<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: Mat<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: Mat<f64> = vec.new_matrix_n_by_m::<0>(Some(M));

    // Expected matrices.
    let mat_n_by_n_exp: Mat<f64> = Mat::new_with_shape(N, N);
    let mat_m_by_n_exp: Mat<f64> = Mat::new_with_shape(M, N);
    let mat_n_by_m_exp: Mat<f64> = Mat::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix_from_dvector() {
    // Vector.
    let vec: DVector<f64> = DVector::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: DMatrix<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: DMatrix<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: DMatrix<f64> = vec.new_matrix_n_by_m::<0>(Some(M));

    // Expected matrices.
    let mat_n_by_n_exp: DMatrix<f64> = DMatrix::new_with_shape(N, N);
    let mat_m_by_n_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_n_by_m_exp: DMatrix<f64> = DMatrix::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_smatrix_from_svector() {
    // Vector.
    let vec: SVector<f64, N> = SVector::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: SMatrix<f64, N, N> = vec.new_matrix_n_by_n();
    let mat_m_by_n: SMatrix<f64, M, N> = vec.new_matrix_m_by_n::<M>(None);
    let mat_n_by_m: SMatrix<f64, N, M> = vec.new_matrix_n_by_m::<M>(None);

    // Expected matrices.
    let mat_n_by_n_exp: SMatrix<f64, N, N> = SMatrix::new_with_shape(N, N);
    let mat_m_by_n_exp: SMatrix<f64, M, N> = SMatrix::new_with_shape(M, N);
    let mat_n_by_m_exp: SMatrix<f64, N, M> = SMatrix::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2_from_array1() {
    // Vector.
    let vec: Array1<f64> = Array1::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: Array2<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: Array2<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: Array2<f64> = vec.new_matrix_n_by_m::<0>(Some(M));

    // Expected matrices.
    let mat_n_by_n_exp: Array2<f64> = Array2::new_with_shape(N, N);
    let mat_m_by_n_exp: Array2<f64> = Array2::new_with_shape(M, N);
    let mat_n_by_m_exp: Array2<f64> = Array2::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(Matrix::shape(&mat_n_by_n), (N, N));
    assert_eq!(Matrix::shape(&mat_m_by_n), (M, N));
    assert_eq!(Matrix::shape(&mat_n_by_m), (N, M));
}
