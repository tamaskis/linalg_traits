use linalg_traits::{Mat, Matrix};
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, Matrix2};

#[cfg(feature = "with_ndarray")]
use ndarray::Array2;

// Test conditions.
static X_ROW: &[f64; 4] = &[1.0, 2.0, 3.0, 4.0];
static Y: f64 = 2.0;

// Expected result.
static Z_ROW: &[f64; 4] = &[2.0, 4.0, 6.0, 8.0];
static Z_COL: &[f64; 4] = &[2.0, 6.0, 4.0, 8.0];

#[test]
fn test_mat() {
    let mut x = Mat::from_row_slice(2, 2, X_ROW);
    let z = x.mul(Y);
    x.mul_assign(Y);
    assert_arrays_equal!(z.as_slice(), Z_ROW);
    assert_arrays_equal!(x.as_slice(), Z_ROW);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix() {
    let mut x = DMatrix::from_row_slice(2, 2, X_ROW);
    let z = x.mul(Y);
    x.mul_assign(Y);
    assert_arrays_equal!(z.as_slice(), Z_COL);
    assert_arrays_equal!(x.as_slice(), Z_COL);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_smatrix() {
    let mut x = <Matrix2<f64> as Matrix<f64>>::from_row_slice(2, 2, X_ROW);
    let z = x.mul(Y);
    x.mul_assign(Y);
    assert_arrays_equal!(z.as_slice(), Z_COL);
    assert_arrays_equal!(x.as_slice(), Z_COL);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2() {
    let mut x = Array2::from_row_slice(2, 2, X_ROW);
    let z = x.mul(Y);
    x.mul_assign(Y);
    assert_arrays_equal!(Matrix::as_slice(&x), Z_ROW);
    assert_arrays_equal!(Matrix::as_slice(&z), Z_ROW);
}
