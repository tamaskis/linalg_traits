use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Test conditions.
static X: &[f64; 3] = &[25.0, 30.0, 35.0];
static Y: &[f64; 3] = &[4.0, 5.0, 6.0];

// Expected result.
static Z: &[f64; 3] = &[21.0, 25.0, 29.0];

// Slice with the wrong length.
static W: &[f64; 2] = &[1.0, 2.0];

#[test]
fn test_vec() {
    let mut x = Vec::from_slice(X);
    let y = Vec::from_slice(Y);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
fn test_vec_sub_panic() {
    let x = Vec::from_slice(X);
    let w = Vec::from_slice(W);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
fn test_vec_sub_assign_panic() {
    let mut x = Vec::from_slice(X);
    let w = Vec::from_slice(W);
    x.sub_assign(&w);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mut x = DVector::from_slice(X);
    let y = DVector::from_slice(Y);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (3, 1)\n right: (2, 1)"
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_sub_panic() {
    let x = DVector::from_slice(X);
    let w = DVector::from_slice(W);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (3, 1)\n right: (2, 1)"
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_sub_assign_panic() {
    let mut x = DVector::from_slice(X);
    let w = DVector::from_slice(W);
    x.sub_assign(&w);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mut x = Vector3::from_slice(X);
    let y = Vector3::from_slice(Y);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mut x = Array1::from_slice(X);
    let y = Array1::from_slice(Y);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(expected = "ShapeError/IncompatibleShape: incompatible shapes")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_sub_panic() {
    let x = Array1::from_slice(X);
    let w = Array1::from_slice(W);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_sub_assign_panic() {
    let mut x = Array1::from_slice(X);
    let w = Array1::from_slice(W);
    x.sub_assign(&w);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let mut x = FMat::from_slice(X);
    let y = FMat::from_slice(Y);
    let z = x.sub(&y);
    x.sub_assign(&y);
    assert_arrays_equal!(z.as_slice(), Z);
    assert_arrays_equal!(x.as_slice(), Z);
}

#[test]
#[should_panic(expected = "Assertion failed: lhs.nrows() == rhs.nrows()")]
#[cfg(feature = "faer")]
fn test_faer_mat_sub_panic() {
    let x = FMat::from_slice(X);
    let w = FMat::from_slice(W);
    let _ = x.sub(&w);
}

#[test]
#[should_panic(expected = "Assertion failed: Head :: nrows(& head) == Tail :: nrows(& tail)")]
#[cfg(feature = "faer")]
fn test_faer_mat_sub_assign_panic() {
    let mut x = FMat::from_slice(X);
    let w = FMat::from_slice(W);
    x.sub_assign(&w);
}
