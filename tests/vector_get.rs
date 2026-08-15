#[cfg(feature = "faer")]
use faer::Col;

use linalg_traits::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

// Test conditions.
static X: &[f64; 3] = &[1.0, 2.0, 3.0];

fn assert_get<V: Vector<f64>>(x: &V) {
    assert_eq!(Vector::get(x, 0), Some(&1.0));
    assert_eq!(Vector::get(x, 1), Some(&2.0));
    assert_eq!(Vector::get(x, 2), Some(&3.0));
    assert_eq!(Vector::get(x, 3), None);
}

#[test]
fn test_vec() {
    let x = Vec::from_slice(X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let x = DVector::from_slice(X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let x = Vector3::from_slice(X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let x = Array1::from_slice(X);
    assert_get(&x);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let x = <Col<f64> as Vector<f64>>::from_slice(X);
    assert_get(&x);
}
