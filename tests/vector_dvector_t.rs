#[cfg(feature = "faer")]
use faer::Col;
use linalg_traits::{RealField, Vector};
#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, SVector, dvector};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, array};
use numtest::*;

// Helper function.
fn dvector_t_test_helper<R: RealField, V: Vector<R>>(x: V) -> V::DVectorT<f64> {
    V::DVectorT::<f64>::new_with_length(x.len())
}

#[test]
fn test_vec() {
    let vec_generic: Vec<f64> = dvector_t_test_helper(Vec::<f64>::new_with_length(3));
    let vec_generic_exp: Vec<f64> = vec![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra() {
    let vec_generic: DVector<f64> = dvector_t_test_helper(DVector::<f64>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = dvector![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);

    let vec_generic: DVector<f64> = dvector_t_test_helper(SVector::<f64, 3>::new_with_length(3));
    let vec_generic_exp: DVector<f64> = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray() {
    let vec_generic: Array1<f64> = dvector_t_test_helper(Array1::<f64>::new_with_length(3));
    let vec_generic_exp: Array1<f64> = array![0.0, 0.0, 0.0];
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}

#[test]
#[cfg(feature = "faer")]
fn test_faer() {
    let vec_generic: Col<f64> = dvector_t_test_helper(Col::<f64>::new_with_length(3));
    let vec_generic_exp: Col<f64> = Col::<f64>::from_fn(3, |_| 0.0);
    assert_arrays_equal!(vec_generic, vec_generic_exp);
}
