use linalg_traits::{Mat, Matrix};
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Slices to use for all tests.
const ROW_SLICE: &[f64; 6] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const COL_SLICE: &[f64; 6] = &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

// Note that `Mat` is row-major.
#[test]
fn test_mat() {
    // Testing from a row slice.
    let x1 = <Mat<f64> as Matrix<f64>>::from_row_slice(2, 3, ROW_SLICE);

    // Testing from a column slice.
    let x2 = <Mat<f64> as Matrix<f64>>::from_col_slice(2, 3, COL_SLICE);

    // Testing equality of the two matrices.
    assert_arrays_equal!(x1, x2);

    // Testing slice representations of x1.
    assert_arrays_equal!(x1.as_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_col_slice(), COL_SLICE);

    // Testing slice representations of x2.
    assert_arrays_equal!(x2.as_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_col_slice(), COL_SLICE);
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
fn test_mat_panic_1() {
    _ = <Mat<f64> as Matrix<f64>>::from_row_slice(2, 2, ROW_SLICE)
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
fn test_mat_panic_2() {
    _ = <Mat<f64> as Matrix<f64>>::from_col_slice(2, 2, COL_SLICE)
}

// Note that `nalgebra::DMatrix` is column-major.
#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    // Testing from a row slice.
    let x1 = <DMatrix<f64> as Matrix<f64>>::from_row_slice(2, 3, ROW_SLICE);

    // Testing from a column slice.
    let x2 = <DMatrix<f64> as Matrix<f64>>::from_col_slice(2, 3, COL_SLICE);

    // Testing equality of the two matrices.
    assert_arrays_equal!(x1, x2);

    // Testing slice representations of x1.
    assert_arrays_equal!(x1.as_slice(), COL_SLICE);
    assert_arrays_equal!(x1.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_col_slice(), COL_SLICE);

    // Testing slice representations of x2.
    assert_arrays_equal!(x2.as_slice(), COL_SLICE);
    assert_arrays_equal!(x2.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_col_slice(), COL_SLICE);
}

#[test]
#[should_panic(
    expected = "Matrix init. error: the slice did not contain the right number of elements."
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix_panic_1() {
    _ = <DMatrix<f64> as Matrix<f64>>::from_row_slice(2, 2, ROW_SLICE)
}

#[test]
#[should_panic(
    expected = "Allocation from iterator error: the iterator did not yield the correct number of elements."
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix_panic_2() {
    _ = <DMatrix<f64> as Matrix<f64>>::from_col_slice(2, 2, COL_SLICE)
}

// Note that `nalgebra::SMatrix` is column-major.
#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    // Testing from a row slice.
    let x1 = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(2, 3, ROW_SLICE);

    // Testing from a column slice.
    let x2 = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_col_slice(2, 3, COL_SLICE);

    // Testing equality of the two matrices.
    assert_arrays_equal!(x1, x2);

    // Testing slice representations of x1.
    assert_arrays_equal!(x1.as_slice(), COL_SLICE);
    assert_arrays_equal!(x1.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_col_slice(), COL_SLICE);

    // Testing slice representations of x2.
    assert_arrays_equal!(x2.as_slice(), COL_SLICE);
    assert_arrays_equal!(x2.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_col_slice(), COL_SLICE);
}

#[test]
#[should_panic(expected = "Column count mismatch.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_panic_1() {
    _ = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(2, 2, ROW_SLICE)
}

#[test]
#[should_panic(expected = "Row count mismatch.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_panic_2() {
    _ = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_row_slice(3, 3, ROW_SLICE)
}

#[test]
#[should_panic(expected = "Column count mismatch.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_panic_3() {
    _ = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_col_slice(2, 2, COL_SLICE)
}

#[test]
#[should_panic(expected = "Row count mismatch.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix_panic_4() {
    _ = <SMatrix<f64, 2, 3> as Matrix<f64>>::from_col_slice(3, 3, COL_SLICE)
}

// Note that `ndarray::Array2` is row-major.
#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    // Testing from a row slice.
    let x1 = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 3, ROW_SLICE);

    // Testing from a column slice.
    let x2 = <Array2<f64> as Matrix<f64>>::from_col_slice(2, 3, COL_SLICE);

    // Testing equality of the two matrices.
    assert_arrays_equal!(x1, x2);

    // Testing slice representations of x1.
    assert_arrays_equal!(<Array2<f64> as Matrix<f64>>::as_slice(&x1), ROW_SLICE);
    assert_arrays_equal!(x1.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_col_slice(), COL_SLICE);

    // Testing slice representations of x2.
    assert_arrays_equal!(<Array2<f64> as Matrix<f64>>::as_slice(&x2), ROW_SLICE);
    assert_arrays_equal!(x2.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_col_slice(), COL_SLICE);
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2_panic_1() {
    _ = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 2, ROW_SLICE)
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2_panic_2() {
    _ = <Array2<f64> as Matrix<f64>>::from_row_slice(2, 2, COL_SLICE)
}

// Note that `faer::Mat` is column-major.
#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    // Testing from a row slice.
    let x1: FMat<f64> = FMat::from_row_slice(2, 3, ROW_SLICE);

    // Testing from a column slice.
    let x2: FMat<f64> = FMat::from_col_slice(2, 3, COL_SLICE);

    // Testing slice representations of x1.
    assert_arrays_equal!(x1.as_slice(), COL_SLICE);
    assert_arrays_equal!(x1.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x1.as_col_slice(), COL_SLICE);

    // Testing slice representations of x2.
    assert_arrays_equal!(x2.as_slice(), COL_SLICE);
    assert_arrays_equal!(x2.as_row_slice(), ROW_SLICE);
    assert_arrays_equal!(x2.as_col_slice(), COL_SLICE);
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
#[cfg(feature = "faer")]
fn test_faer_mat_panic_1() {
    _ = FMat::<f64>::from_row_slice(2, 2, ROW_SLICE)
}

#[test]
#[should_panic(expected = "Slice length (6) not compatible with matrix dimensions (2x2).")]
#[cfg(feature = "faer")]
fn test_faer_mat_panic_2() {
    _ = FMat::<f64>::from_col_slice(2, 2, COL_SLICE)
}
