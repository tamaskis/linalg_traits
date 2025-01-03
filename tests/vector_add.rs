use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

// Test conditions.
static X: &[f64; 3] = &[1.0, 2.0, 3.0];
static Y: &[f64; 3] = &[4.0, 5.0, 6.0];

// Expected result.
static Z: &[f64; 3] = &[5.0, 7.0, 9.0];

// Slice with the wrong length.
static W: &[f64; 2] = &[1.0, 2.0];

#[test]
fn test_vec() {
    let mut x = Vec::from_slice(X);
    let y = Vec::from_slice(Y);
    let z = x.add(&y);
    x.add_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
fn test_vec_add_panic() {
    let x = Vec::from_slice(X);
    let w = Vec::from_slice(W);
    let _ = x.add(&w);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
fn test_vec_add_assign_panic() {
    let mut x = Vec::from_slice(X);
    let w = Vec::from_slice(W);
    x.add_assign(&w);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mut x = DVector::from_slice(X);
    let y = DVector::from_slice(Y);
    let z = x.add(&y);
    x.add_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (3, 1)\n right: (2, 1)"
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_add_panic() {
    let x = DVector::from_slice(X);
    let w = DVector::from_slice(W);
    let _ = x.add(&w);
}

#[test]
#[should_panic(
    expected = "Matrix addition/subtraction dimensions mismatch.\n  left: (3, 1)\n right: (2, 1)"
)]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_add_assign_panic() {
    let mut x = DVector::from_slice(X);
    let w = DVector::from_slice(W);
    x.add_assign(&w);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mut x = Vector3::from_slice(X);
    let y = Vector3::from_slice(Y);
    let z = x.add(&y);
    x.add_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mut x = Array1::from_slice(X);
    let y = Array1::from_slice(Y);
    let z = x.add(&y);
    x.add_assign(&y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[should_panic(expected = "ShapeError/IncompatibleShape: incompatible shapes")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_add_panic() {
    let x = Array1::from_slice(X);
    let w = Array1::from_slice(W);
    let _ = x.add(&w);
}

#[test]
#[should_panic(
    expected = "Length of the other vector (3) does not match the length of this vector (2)."
)]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_add_assign_panic() {
    let mut x = Array1::from_slice(X);
    let w = Array1::from_slice(W);
    x.add_assign(&w);
}
