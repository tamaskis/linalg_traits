#[cfg(feature = "faer")]
use faer::Mat as FMat;
use linalg_traits::Matrix;
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};
#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[test]
fn test_mat() {
    let mut x = <linalg_traits::Mat<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 0)] = 1.0;
    x[(0, 1)] = 2.0;
    x[(0, 2)] = 3.0;
    x[(1, 0)] = 4.0;
    x[(1, 1)] = 5.0;
    x[(1, 2)] = 6.0;
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
    let mut x = <linalg_traits::Mat<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(2, 0)] = 4.0;
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    let mut x = <DMatrix<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 0)] = 1.0;
    x[(0, 1)] = 2.0;
    x[(0, 2)] = 3.0;
    x[(1, 0)] = 4.0;
    x[(1, 1)] = 5.0;
    x[(1, 2)] = 6.0;
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
    let mut x = <DMatrix<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 3)] = 4.0;
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    let mut x = <SMatrix<f64, 2, 3> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 0)] = 1.0;
    x[(0, 1)] = 2.0;
    x[(0, 2)] = 3.0;
    x[(1, 0)] = 4.0;
    x[(1, 1)] = 5.0;
    x[(1, 2)] = 6.0;
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
    let mut x = <SMatrix<f64, 2, 3> as Matrix<f64>>::new_with_shape(2, 3);
    x[(2, 0)] = 4.0;
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    let mut x = <Array2<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 0)] = 1.0;
    x[(0, 1)] = 2.0;
    x[(0, 2)] = 3.0;
    x[(1, 0)] = 4.0;
    x[(1, 1)] = 5.0;
    x[(1, 2)] = 6.0;
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
    let mut x = <Array2<f64> as Matrix<f64>>::new_with_shape(2, 3);
    x[(0, 3)] = 4.0;
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let mut x = FMat::<f64>::new_with_shape(2, 3);
    x[(0, 0)] = 1.0;
    x[(0, 1)] = 2.0;
    x[(0, 2)] = 3.0;
    x[(1, 0)] = 4.0;
    x[(1, 1)] = 5.0;
    x[(1, 2)] = 6.0;
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
    let mut x = FMat::<f64>::new_with_shape(2, 3);
    x[(2, 0)] = 4.0;
}
