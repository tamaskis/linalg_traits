use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

// Slice to use for all tests.
const X: &[f64; 3] = &[1.0, 2.0, 3.0];

#[test]
fn test_vec() {
    let x = <Vec<f64> as Vector>::from_slice(X);
    assert_arrays_equal!(x, X);
    assert_arrays_equal!(<Vec<f64> as Vector>::as_slice(&x), X);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dvector() {
    let x = <DVector<f64> as Vector>::from_slice(X);
    assert_arrays_equal!(x, X);
    assert_arrays_equal!(<DVector<f64> as Vector>::as_slice(&x), X);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_svector() {
    let x = <SVector<f64, 3> as Vector>::from_slice(X);
    assert_arrays_equal!(x, X);
    assert_arrays_equal!(<SVector<f64, 3> as Vector>::as_slice(&x), X);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array1() {
    let x = <Array1<f64> as Vector>::from_slice(X);
    assert_arrays_equal!(x, X);
    assert_arrays_equal!(<Array1<f64> as Vector>::as_slice(&x), X);
}
