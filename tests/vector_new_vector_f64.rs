#[cfg(feature = "faer")]
use faer::Col;
use linalg_traits::Vector;
#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, SVector, dvector};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, array};
use numtest::*;

#[test]
fn test_vec() {
    let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
    let vec_f64: Vec<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra() {
    let vec: DVector<f64> = dvector![1.0, 2.0, 3.0];
    let vec_f64: DVector<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);

    let vec: SVector<f64, 3> = SVector::from_slice(&[1.0, 2.0, 3.0]);
    let vec_f64: SVector<f64, 3> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray() {
    let vec: Array1<f64> = array![1.0, 2.0, 3.0];
    let vec_f64: Array1<f64> = vec.new_vector_f64();
    assert_arrays_equal!(vec_f64, [0.0, 0.0, 0.0]);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer() {
    let vec: Col<f64> = Col::from_slice(&[1.0, 2.0, 3.0]);
    let vec_f64: Col<f64> = vec.new_vector_f64();
    assert_arrays_equal!(Vector::as_slice(&vec_f64), [0.0, 0.0, 0.0]);
}
