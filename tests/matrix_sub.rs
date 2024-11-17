use linalg_traits::{Mat, Matrix};
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, Matrix2};

#[cfg(feature = "with_ndarray")]
use ndarray::Array2;

// Test conditions.
static X_ROW: &[f64; 4] = &[1.0, 2.0, 3.0, 4.0];
static Y_ROW: &[f64; 4] = &[10.0, 7.0, 4.0, 1.0];

// Expected result.
static Z_ROW: &[f64; 4] = &[-9.0, -5.0, -1.0, 3.0];
static Z_COL: &[f64; 4] = &[-9.0, -1.0, -5.0, 3.0];

// Slice with the wrong length.
static W_ROW: &[f64; 6] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn test_mat() {
    let mut x = Mat::from_row_slice(2, 2, X_ROW);
    let y = Mat::from_row_slice(2, 2, Y_ROW);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z.as_slice(), Z_ROW);
    assert_arrays_equal!(x.as_slice(), Z_ROW);
}

#[test]
#[should_panic(expected = " Matrices have incompatible shapes.\n  left: (2, 2)\n right: (2, 3)")]
fn test_mat_sub_panic() {
    let x = Mat::from_row_slice(2, 2, X_ROW);
    let w = Mat::from_row_slice(2, 3, W_ROW);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(expected = " Matrices have incompatible shapes.\n  left: (2, 2)\n right: (2, 3)")]
fn test_mat_sub_assign_panic() {
    let mut x = Mat::from_row_slice(2, 2, X_ROW);
    let w = Mat::from_row_slice(2, 3, W_ROW);
    x.sub_assign(&w);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix() {
    let mut x = DMatrix::from_row_slice(2, 2, X_ROW);
    let y = DMatrix::from_row_slice(2, 2, Y_ROW);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z.as_slice(), Z_COL);
    assert_arrays_equal!(x.as_slice(), Z_COL);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (2, 2)\n right: (2, 3)"
)]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix_sub_panic() {
    let x = DMatrix::from_row_slice(2, 2, X_ROW);
    let w = DMatrix::from_row_slice(2, 3, W_ROW);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (2, 2)\n right: (2, 3)"
)]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix_sub_assign_panic() {
    let mut x = DMatrix::from_row_slice(2, 2, X_ROW);
    let w = DMatrix::from_row_slice(2, 3, W_ROW);
    x.sub_assign(&w);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_smatrix() {
    let mut x = <Matrix2<f64> as Matrix<f64>>::from_row_slice(2, 2, X_ROW);
    let y = <Matrix2<f64> as Matrix<f64>>::from_row_slice(2, 2, Y_ROW);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z.as_slice(), Z_COL);
    assert_arrays_equal!(x.as_slice(), Z_COL);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2() {
    let mut x = Array2::from_row_slice(2, 2, X_ROW);
    let y = Array2::from_row_slice(2, 2, Y_ROW);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(Matrix::as_slice(&x), Z_ROW);
    assert_arrays_equal!(Matrix::as_slice(&z), Z_ROW);
}

#[test]
#[should_panic(expected = "ShapeError/IncompatibleShape: incompatible shapes")]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2_sub_panic() {
    let x = Array2::from_row_slice(2, 2, X_ROW);
    let w = Array2::from_row_slice(2, 3, W_ROW);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(expected = "ndarray: could not broadcast array from shape: [2, 3] to: [2, 2]")]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2_sub_assign_panic() {
    let mut x = Array2::from_row_slice(2, 2, X_ROW);
    let w = Array2::from_row_slice(2, 3, W_ROW);
    x.sub_assign(&w);
}
