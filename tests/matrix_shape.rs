use linalg_traits::{Mat, Matrix};

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[test]
fn test_mat() {
    assert_eq!(
        <Mat<f64> as Matrix<f64>>::new_with_shape(3, 2).shape(),
        (3, 2)
    );
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    assert_eq!(
        <DMatrix<f64> as Matrix<f64>>::new_with_shape(3, 2).shape(),
        (3, 2)
    );
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    assert_eq!(
        <SMatrix<f64, 3, 2> as Matrix<f64>>::new_with_shape(3, 2).shape(),
        (3, 2)
    );
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    let matrix = <Array2<f64> as Matrix<f64>>::new_with_shape(3, 2);
    assert_eq!(Matrix::shape(&matrix), (3, 2));
}
