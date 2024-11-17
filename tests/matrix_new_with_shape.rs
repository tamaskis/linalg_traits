use linalg_traits::{Mat, Matrix};
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "with_ndarray")]
use ndarray::Array2;

#[test]
fn test_new_with_shape_vec() {
    assert_arrays_equal!(
        <Mat<f64> as Matrix<f64>>::new_with_shape(3, 2),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_dmatrix() {
    assert_arrays_equal!(
        <DMatrix<f64> as Matrix<f64>>::new_with_shape(3, 2),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_smatrix() {
    assert_arrays_equal!(
        <SMatrix<f64, 3, 2> as Matrix<f64>>::new_with_shape(3, 2),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "Row count mismatch.\n  left: 2\n right: 3")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_smatrix_panic_1() {
    let _ = <SMatrix<f64, 3, 2> as Matrix<f64>>::new_with_shape(2, 3);
}

#[test]
#[should_panic(expected = "Row count mismatch.\n  left: 3\n right: 2")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_smatrix_panic_2() {
    let _ = <SMatrix<f64, 2, 3> as Matrix<f64>>::new_with_shape(3, 2);
}

#[test]
#[should_panic(expected = "Row count mismatch.\n  left: 2\n right: 3")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_smatrix_panic_3() {
    let _ = <SMatrix<f64, 3, 3> as Matrix<f64>>::new_with_shape(2, 3);
}

#[test]
#[should_panic(expected = "Column count mismatch.\n  left: 2\n right: 3")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_shape_nalgebra_smatrix_panic_4() {
    let _ = <SMatrix<f64, 3, 3> as Matrix<f64>>::new_with_shape(3, 2);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_new_with_shape_ndarray_array2() {
    assert_arrays_equal!(
        <Array2<f64> as Matrix<f64>>::new_with_shape(3, 2),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}
