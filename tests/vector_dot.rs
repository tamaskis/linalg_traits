use linalg_traits::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

// Test conditions.
static X: &[f64; 3] = &[1.0, 2.0, 3.0];
static Y: &[f64; 3] = &[4.0, 5.0, 6.0];

// Expected result.
static Z: f64 = 32.0;

// Slice with the wrong length.
static W: &[f64; 2] = &[1.0, 2.0];

#[test]
fn test_vec() {
    let x = Vec::from_slice(X);
    let y = Vec::from_slice(Y);
    let z = x.dot(&y);
    assert_eq!(z, Z);
}

#[test]
#[should_panic(expected = "Cannot evaluate the dot product of vectors with different lengths.")]
fn test_vec_panic() {
    let x = Vec::from_slice(X);
    let w = Vec::from_slice(W);
    let _ = x.dot(&w);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dvector() {
    let x = DVector::from_slice(X);
    let y = DVector::from_slice(Y);
    let z = x.dot(&y);
    assert_eq!(z, Z);
}

#[test]
#[should_panic(expected = "Dot product dimensions mismatch for shapes (3, 1) and (2, 1)")]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dvector_panic() {
    let x = DVector::from_slice(X);
    let w = DVector::from_slice(W);
    let _ = x.dot(&w);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_svector() {
    let x = Vector3::from_slice(X);
    let y = Vector3::from_slice(Y);
    let z = x.dot(&y);
    assert_eq!(z, Z);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.\n  left: 2\n right: 3")]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_svector_panic() {
    let x = Vector3::from_slice(X);
    let w = Vector3::from_slice(W);
    let _ = x.dot(&w);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array1() {
    let x = Array1::from_slice(X);
    let y = Array1::from_slice(Y);
    let z = x.dot(&y);
    assert_eq!(z, Z);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed\n  left: 3\n right: 2")]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array1_panic() {
    let x = Array1::from_slice(X);
    let w = Array1::from_slice(W);
    let _ = x.dot(&w);
}
