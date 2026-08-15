#[cfg(feature = "faer")]
use faer::Mat as FMat;
use linalg_traits::Matrix;
#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};
#[cfg(feature = "ndarray")]
use ndarray::Array2;

// Test conditions.
static X: &[f64; 6] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

fn assert_get<M: Matrix<f64>>(x: &M) {
    assert_eq!(Matrix::get(x, (0, 0)), Some(&1.0));
    assert_eq!(Matrix::get(x, (0, 1)), Some(&2.0));
    assert_eq!(Matrix::get(x, (0, 2)), Some(&3.0));
    assert_eq!(Matrix::get(x, (1, 0)), Some(&4.0));
    assert_eq!(Matrix::get(x, (1, 1)), Some(&5.0));
    assert_eq!(Matrix::get(x, (1, 2)), Some(&6.0));
    assert_eq!(Matrix::get(x, (2, 0)), None);
}

#[test]
fn test_mat() {
    let x = <linalg_traits::Mat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    let x = <DMatrix<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    let x = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    let x = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let x = <FMat<f64> as Matrix<f64>>::from_row_slice(2, 3, X);
    assert_get(&x);
}
