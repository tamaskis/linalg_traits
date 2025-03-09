use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector3};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

// Test conditions.
static X: &[f64; 3] = &[1.0, 2.0, 3.0];

#[test]
fn test_vec() {
    let mut x = Vec::new_with_length(3);
    x.vset(0, 1.0);
    x.vset(1, 2.0);
    x.vset(2, 3.0);
    assert_arrays_equal!(x, X);
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
fn test_vec_out_of_bounds() {
    let mut x = Vec::new_with_length(3);
    x.vset(3, 4.0);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    let mut x = DVector::new_with_length(3);
    x.vset(0, 1.0);
    x.vset(1, 2.0);
    x.vset(2, 3.0);
    assert_arrays_equal!(x, X);
}

#[test]
#[should_panic(expected = "Matrix index out of bounds.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector_out_of_bounds() {
    let mut x = DVector::new_with_length(3);
    x.vset(3, 4.0);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    let mut x = Vector3::new_with_length(3);
    x.vset(0, 1.0);
    x.vset(1, 2.0);
    x.vset(2, 3.0);
    assert_arrays_equal!(x, X);
}

#[test]
#[should_panic(expected = "Matrix index out of bounds.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector_out_of_bounds() {
    let mut x = Vector3::new_with_length(3);
    x.vset(3, 4.0);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    let mut x = Array1::new_with_length(3);
    x.vset(0, 1.0);
    x.vset(1, 2.0);
    x.vset(2, 3.0);
    assert_arrays_equal!(x, X);
}

#[test]
#[should_panic(expected = "ndarray: index 3 is out of bounds for array of shape [3]")]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1_out_of_bounds() {
    let mut x = Array1::new_with_length(3);
    x.vset(3, 4.0);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    let mut x = FMat::new_with_length(3);
    x.vset(0, 1.0);
    x.vset(1, 2.0);
    x.vset(2, 3.0);
    assert_arrays_equal!(x.as_slice(), X);
}

#[test]
#[should_panic(expected = "Assertion failed: row < self.nrows()")]
#[cfg(feature = "faer")]
fn test_faer_mat_out_of_bounds() {
    let mut x = FMat::new_with_length(3);
    x.vset(3, 4.0);
}
