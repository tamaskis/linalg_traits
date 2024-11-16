use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[test]
fn test_new_with_length_vec() {
    assert_arrays_equal!(
        <Vec<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_nalgebra_dvector() {
    assert_arrays_equal!(
        <DVector<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_nalgebra_svector() {
    assert_arrays_equal!(
        <SVector<f64, 3> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_nalgebra_svector_panic() {
    let _ = <SVector<f64, 2> as Vector<f64>>::new_with_length(3);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_new_with_length_ndarray_array1() {
    assert_arrays_equal!(
        <Array1<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}
