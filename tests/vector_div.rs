use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Test conditions.
static X: &[f64; 3] = &[100.0, 200.0, 300.0];
static Y: f64 = 10.0;

// Expected result.
static Z: &[f64; 3] = &[10.0, 20.0, 30.0];

#[test]
fn test_vec() {
    let mut x = Vec::from_slice(X);
    let z = x.div(Y);
    x.div_assign(Y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mut x = DVector::from_slice(X);
    let z = x.div(Y);
    x.div_assign(Y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mut x = Vector3::from_slice(X);
    let z = x.div(Y);
    x.div_assign(Y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mut x = Array1::from_slice(X);
    let z = x.div(Y);
    x.div_assign(Y);
    assert_arrays_equal!(z, Z);
    assert_arrays_equal!(x, Z);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let mut x = FMat::from_slice(X);
    let z = x.div(Y);
    x.div_assign(Y);
    assert_arrays_equal!(z.as_slice(), Z);
    assert_arrays_equal!(x.as_slice(), Z);
}
