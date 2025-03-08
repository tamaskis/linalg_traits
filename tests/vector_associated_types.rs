use linalg_traits::{Mat, Matrix, Scalar, Vector};
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{dvector, DMatrix, DVector, SMatrix, SVector};

#[cfg(feature = "ndarray")]
use ndarray::{array, Array1, Array2};

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Dimensions for all unit tests.
const M: usize = 3;
const N: usize = 2;

#[test]
#[cfg(feature = "nalgebra")]
#[cfg(feature = "ndarray")]
#[cfg(feature = "faer")]
fn test_vector_t() {
    // Helper function.
    fn vector_t_test_helper<S: Scalar, V: Vector<S>>(x: V) -> V::VectorT<f64> {
        V::VectorT::<f64>::new_with_length(x.len())
    }

    // Vec<f64> from Vec<f64>.
    let vec_generic: Vec<f64> = vector_t_test_helper(Vec::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::DVector<f64> from nalgebra::DVector<f64>.
    let vec_generic: DVector<f64> = vector_t_test_helper(DVector::<f64>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = dvector![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::SVector<f64, 3> from nalgebra::SVector<f64, 3>.
    let vec_generic: SVector<f64, 3> = vector_t_test_helper(SVector::<f64, 3>::new_with_length(3));
    let vec_generic_exp: SVector<f64, 3> = SVector::from_row_slice(&[0.0, 0.0, 0.0]);
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // Vec<f64> from ndarray::Array1<f64>.
    let vec_generic: Vec<f64> = vector_t_test_helper(Array1::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // Vec<f64> from faer::Mat<f64>.
    let vec_generic: Vec<f64> = vector_t_test_helper(FMat::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}

#[test]
#[cfg(feature = "nalgebra")]
#[cfg(feature = "ndarray")]
#[cfg(feature = "faer")]
fn test_dvector_t() {
    // Helper function.
    fn dvector_t_test_helper<S: Scalar, V: Vector<S>>(x: V) -> V::DVectorT<f64> {
        V::DVectorT::<f64>::new_with_length(x.len())
    }

    // Vec<f64> from Vec<f64>.
    let vec_generic: Vec<f64> = dvector_t_test_helper(Vec::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::DVector<f64> nalgebra::from DVector<f64>.
    let vec_generic: DVector<f64> = dvector_t_test_helper(DVector::<f64>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = dvector![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::SVector<f64, 3> from nalgebra::SVector<f64, 3>.
    let vec_generic: DVector<f64> = dvector_t_test_helper(SVector::<f64, 3>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // Vec<f64> from ndarray::Array1<f64>.
    let vec_generic: Vec<f64> = dvector_t_test_helper(Array1::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // Vec<f64> from faer::Mat<f64>.
    let vec_generic: Vec<f64> = dvector_t_test_helper(FMat::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}

#[test]
#[cfg(feature = "nalgebra")]
#[cfg(feature = "ndarray")]
#[cfg(feature = "faer")]
fn test_dvector_f64() {
    // Helper function.
    fn dvector_f64_test_helper<S: Scalar, V: Vector<S>>(x: V) -> V::DVectorf64 {
        V::DVectorf64::new_with_length(x.len())
    }

    // Vec<f64> from Vec<f64>.
    let vec_generic: Vec<f64> = dvector_f64_test_helper(Vec::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::DVector<f64> from nalgebra::DVector<f64>.
    let vec_generic: DVector<f64> = dvector_f64_test_helper(DVector::<f64>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = dvector![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // nalgebra::SVector<f64, 3> from nalgebra::SVector<f64, 3>.
    let vec_generic: DVector<f64> = dvector_f64_test_helper(SVector::<f64, 3>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // ndarray::Array1<f64> from ndarray::Array1<f64>.
    let vec_generic: Array1<f64> = dvector_f64_test_helper(Array1::<f64>::new_with_length(3));
    let vec_generic_exp: Array1<f64> = array![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    // faer::Mat<f64> from faer::Mat<f64>.
    let vec_generic: FMat<f64> = dvector_f64_test_helper(FMat::<f64>::new_with_length(3));
    let vec_generic_exp: FMat<f64> = FMat::zeros(3, 1);
    assert_arrays_equal!(
        Vector::as_slice(&vec_generic),
        Vector::as_slice(&vec_generic_exp)
    );
}

#[test]
#[cfg(feature = "nalgebra")]
#[cfg(feature = "ndarray")]
#[cfg(feature = "faer")]
fn test_new_vector_f64() {
    // Vec<f64> from Vec<f64>.
    let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
    let vec_f64: Vec<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);

    // nalgebra::DVector<f64> from nalgebra::DVector<f64>.
    let vec: DVector<f64> = dvector![1.0, 2.0, 3.0];
    let vec_f64: DVector<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);

    // nalgebra::SVector<f64, 3> from nalgebra::SVector<f64, 3>.
    let vec: SVector<f64, 3> = SVector::from_slice(&[1.0, 2.0, 3.0]);
    let vec_f64: SVector<f64, 3> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);

    // ndarray::Array1<f64> from ndarray::Array1<f64>.
    let vec: Array1<f64> = array![1.0, 2.0, 3.0];
    let vec_f64: Array1<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);

    // faer::Mat<f64> from faer::Mat<f64>.
    let vec: FMat<f64> = FMat::from_slice(&[1.0, 2.0, 3.0]);
    let vec_f64: FMat<f64> = vec.new_vector_f64();
    assert_arrays_equal!(Vector::as_slice(&vec_f64), [0.0, 0.0, 0.0]);
}

#[test]
fn test_mat_from_vec() {
    // Vector.
    let vec: Vec<f64> = Vec::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: Mat<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: Mat<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: Mat<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let mat_m_by_n_dynamic: Mat<f64> = vec.new_dmatrix_m_by_n(M);
    let mat_m_by_n_dynamic_f64: Mat<f64> = vec.new_dmatrix_m_by_n_f64(M);
    let mat_n_by_m_dynamic: Mat<f64> = vec.new_dmatrix_n_by_m(M);

    // Expected matrices.
    let mat_n_by_n_exp: Mat<f64> = Mat::new_with_shape(N, N);
    let mat_m_by_n_exp: Mat<f64> = Mat::new_with_shape(M, N);
    let mat_n_by_m_exp: Mat<f64> = Mat::new_with_shape(N, M);
    let mat_m_by_n_dynamic_exp: Mat<f64> = Mat::new_with_shape(M, N);
    let mat_m_by_n_dynamic_f64_exp: Mat<f64> = Mat::new_with_shape(M, N);
    let mat_n_by_m_dynamic_exp: Mat<f64> = Mat::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic, mat_m_by_n_dynamic_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic_f64, mat_m_by_n_dynamic_f64_exp);
    assert_arrays_equal!(mat_n_by_m_dynamic, mat_n_by_m_dynamic_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
    assert_eq!(mat_m_by_n_dynamic.shape(), (M, N));
    assert_eq!(mat_m_by_n_dynamic_f64.shape(), (M, N));
    assert_eq!(mat_n_by_m_dynamic.shape(), (N, M));
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix_from_dvector() {
    // Vector.
    let vec: DVector<f64> = DVector::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: DMatrix<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: DMatrix<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: DMatrix<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let mat_m_by_n_dynamic: DMatrix<f64> = vec.new_dmatrix_m_by_n(M);
    let mat_m_by_n_dynamic_f64: DMatrix<f64> = vec.new_dmatrix_m_by_n_f64(M);
    let mat_n_by_m_dynamic: DMatrix<f64> = vec.new_dmatrix_n_by_m(M);

    // Expected matrices.
    let mat_n_by_n_exp: DMatrix<f64> = DMatrix::new_with_shape(N, N);
    let mat_m_by_n_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_n_by_m_exp: DMatrix<f64> = DMatrix::new_with_shape(N, M);
    let mat_m_by_n_dynamic_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_m_by_n_dynamic_f64_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_n_by_m_dynamic_exp: DMatrix<f64> = DMatrix::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic, mat_m_by_n_dynamic_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic_f64, mat_m_by_n_dynamic_f64_exp);
    assert_arrays_equal!(mat_n_by_m_dynamic, mat_n_by_m_dynamic_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
    assert_eq!(mat_m_by_n_dynamic.shape(), (M, N));
    assert_eq!(mat_m_by_n_dynamic_f64.shape(), (M, N));
    assert_eq!(mat_n_by_m_dynamic.shape(), (N, M));
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_from_svector() {
    // Vector.
    let vec: SVector<f64, N> = SVector::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: SMatrix<f64, N, N> = vec.new_matrix_n_by_n();
    let mat_m_by_n: SMatrix<f64, M, N> = vec.new_matrix_m_by_n::<M>(None);
    let mat_n_by_m: SMatrix<f64, N, M> = vec.new_matrix_n_by_m::<M>(None);
    let mat_m_by_n_dynamic: DMatrix<f64> = vec.new_dmatrix_m_by_n(M);
    let mat_m_by_n_dynamic_f64: DMatrix<f64> = vec.new_dmatrix_m_by_n_f64(M);
    let mat_n_by_m_dynamic: DMatrix<f64> = vec.new_dmatrix_n_by_m(M);

    // Expected matrices.
    let mat_n_by_n_exp: SMatrix<f64, N, N> = SMatrix::new_with_shape(N, N);
    let mat_m_by_n_exp: SMatrix<f64, M, N> = SMatrix::new_with_shape(M, N);
    let mat_n_by_m_exp: SMatrix<f64, N, M> = SMatrix::new_with_shape(N, M);
    let mat_m_by_n_dynamic_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_m_by_n_dynamic_f64_exp: DMatrix<f64> = DMatrix::new_with_shape(M, N);
    let mat_n_by_m_dynamic_exp: DMatrix<f64> = DMatrix::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic, mat_m_by_n_dynamic_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic_f64, mat_m_by_n_dynamic_f64_exp);
    assert_arrays_equal!(mat_n_by_m_dynamic, mat_n_by_m_dynamic_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(mat_n_by_n.shape(), (N, N));
    assert_eq!(mat_m_by_n.shape(), (M, N));
    assert_eq!(mat_n_by_m.shape(), (N, M));
    assert_eq!(mat_m_by_n_dynamic.shape(), (M, N));
    assert_eq!(mat_m_by_n_dynamic_f64.shape(), (M, N));
    assert_eq!(mat_n_by_m_dynamic.shape(), (N, M));
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2_from_array1() {
    // Vector.
    let vec: Array1<f64> = Array1::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: Array2<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: Array2<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: Array2<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let mat_m_by_n_dynamic: Array2<f64> = vec.new_dmatrix_m_by_n(M);
    let mat_m_by_n_dynamic_f64: Array2<f64> = vec.new_dmatrix_m_by_n_f64(M);
    let mat_n_by_m_dynamic: Array2<f64> = vec.new_dmatrix_n_by_m(M);

    // Expected matrices.
    let mat_n_by_n_exp: Array2<f64> = Array2::new_with_shape(N, N);
    let mat_m_by_n_exp: Array2<f64> = Array2::new_with_shape(M, N);
    let mat_n_by_m_exp: Array2<f64> = Array2::new_with_shape(N, M);
    let mat_m_by_n_dynamic_exp: Array2<f64> = Array2::new_with_shape(M, N);
    let mat_m_by_n_dynamic_f64_exp: Array2<f64> = Array2::new_with_shape(M, N);
    let mat_n_by_m_dynamic_exp: Array2<f64> = Array2::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(mat_n_by_n, mat_n_by_n_exp);
    assert_arrays_equal!(mat_m_by_n, mat_m_by_n_exp);
    assert_arrays_equal!(mat_n_by_m, mat_n_by_m_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic, mat_m_by_n_dynamic_exp);
    assert_arrays_equal!(mat_m_by_n_dynamic_f64, mat_m_by_n_dynamic_f64_exp);
    assert_arrays_equal!(mat_n_by_m_dynamic, mat_n_by_m_dynamic_exp);

    // Check the shapes of the actual matrices.
    assert_eq!(Matrix::shape(&mat_n_by_n), (N, N));
    assert_eq!(Matrix::shape(&mat_m_by_n), (M, N));
    assert_eq!(Matrix::shape(&mat_n_by_m), (N, M));
    assert_eq!(Matrix::shape(&mat_m_by_n_dynamic), (M, N));
    assert_eq!(Matrix::shape(&mat_m_by_n_dynamic_f64), (M, N));
    assert_eq!(Matrix::shape(&mat_n_by_m_dynamic), (N, M));
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat_from_faer_mat() {
    // Vector.
    let vec: FMat<f64> = FMat::new_with_length(N);

    // Matrices constructed using `new_matrix_*_by_*`.
    let mat_n_by_n: FMat<f64> = vec.new_matrix_n_by_n();
    let mat_m_by_n: FMat<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    let mat_n_by_m: FMat<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    let mat_m_by_n_dynamic: FMat<f64> = vec.new_dmatrix_m_by_n(M);
    let mat_m_by_n_dynamic_f64: FMat<f64> = vec.new_dmatrix_m_by_n_f64(M);
    let mat_n_by_m_dynamic: FMat<f64> = vec.new_dmatrix_n_by_m(M);

    // Expected matrices.
    let mat_n_by_n_exp: FMat<f64> = FMat::new_with_shape(N, N);
    let mat_m_by_n_exp: FMat<f64> = FMat::new_with_shape(M, N);
    let mat_n_by_m_exp: FMat<f64> = FMat::new_with_shape(N, M);
    let mat_m_by_n_dynamic_exp: FMat<f64> = FMat::new_with_shape(M, N);
    let mat_m_by_n_dynamic_f64_exp: FMat<f64> = FMat::new_with_shape(M, N);
    let mat_n_by_m_dynamic_exp: FMat<f64> = FMat::new_with_shape(N, M);

    // Check equality of elements between actual and expected matrices.
    assert_arrays_equal!(
        Matrix::as_slice(&mat_n_by_n),
        Matrix::as_slice(&mat_n_by_n_exp)
    );
    assert_arrays_equal!(
        Matrix::as_slice(&mat_m_by_n),
        Matrix::as_slice(&mat_m_by_n_exp)
    );
    assert_arrays_equal!(
        Matrix::as_slice(&mat_n_by_m),
        Matrix::as_slice(&mat_n_by_m_exp)
    );
    assert_arrays_equal!(
        Matrix::as_slice(&mat_m_by_n_dynamic),
        Matrix::as_slice(&mat_m_by_n_dynamic_exp)
    );
    assert_arrays_equal!(
        Matrix::as_slice(&mat_m_by_n_dynamic_f64),
        Matrix::as_slice(&mat_m_by_n_dynamic_f64_exp)
    );
    assert_arrays_equal!(
        Matrix::as_slice(&mat_n_by_m_dynamic),
        Matrix::as_slice(&mat_n_by_m_dynamic_exp)
    );

    // Check the shapes of the actual matrices.
    assert_eq!(Matrix::shape(&mat_n_by_n), (N, N));
    assert_eq!(Matrix::shape(&mat_m_by_n), (M, N));
    assert_eq!(Matrix::shape(&mat_n_by_m), (N, M));
    assert_eq!(Matrix::shape(&mat_m_by_n_dynamic), (M, N));
    assert_eq!(Matrix::shape(&mat_m_by_n_dynamic_f64), (M, N));
    assert_eq!(Matrix::shape(&mat_n_by_m_dynamic), (N, M));
}
