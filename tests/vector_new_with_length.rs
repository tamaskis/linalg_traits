use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

#[test]
fn test_vec() {
    assert_arrays_equal!(
        <Vec<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    assert_arrays_equal!(
        <DVector<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    assert_arrays_equal!(
        <SVector<f64, 3> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector_panic() {
    let _ = <SVector<f64, 2> as Vector<f64>>::new_with_length(3);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    assert_arrays_equal!(
        <Array1<f64> as Vector<f64>>::new_with_length(3),
        [0.0, 0.0, 0.0]
    );
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    assert_arrays_equal!(FMat::<f64>::new_with_length(3).as_slice(), [0.0, 0.0, 0.0]);
}
