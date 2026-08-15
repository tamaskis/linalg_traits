#[cfg(feature = "faer")]
use faer::Col;
use linalg_traits::Vector;
#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};
#[cfg(feature = "ndarray")]
use ndarray::Array1;

// Test conditions.
static X: &[f64; 3] = &[1.0, 2.0, 3.0];

#[test]
fn test_vec() {
    let x = Vec::from_slice(X);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(x[2], 3.0);
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
fn test_vec_out_of_bounds() {
    let x = Vec::from_slice(X);
    _ = x[3];
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let x = DVector::from_slice(X);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(x[2], 3.0);
}

#[test]
#[should_panic(expected = "Matrix index out of bounds.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_out_of_bounds() {
    let x = DVector::from_slice(X);
    _ = x[3];
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let x = Vector3::from_slice(X);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(x[2], 3.0);
}

#[test]
#[should_panic(expected = "Matrix index out of bounds.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector_out_of_bounds() {
    let x = Vector3::from_slice(X);
    _ = x[3];
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let x = Array1::from_slice(X);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(x[2], 3.0);
}

#[test]
#[should_panic(expected = "ndarray: index 3 is out of bounds for array of shape [3]")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_out_of_bounds() {
    let x = Array1::from_slice(X);
    _ = x[3];
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let x = <Col<f64> as Vector<f64>>::from_slice(X);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(x[2], 3.0);
}

#[test]
#[should_panic(expected = "Assertion failed: row < self.nrows()")]
#[cfg(feature = "faer")]
fn test_faer_mat_out_of_bounds() {
    let x = <Col<f64> as Vector<f64>>::from_slice(X);
    _ = x[3];
}
