use linalg_traits::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[test]
fn test_is_empty_vec() {
    assert!(<Vec<f64> as Vector<f64>>::new_with_length(0).is_empty());
    assert!(!<Vec<f64> as Vector<f64>>::new_with_length(3).is_empty());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_is_empty_nalgebra_dvector() {
    assert!(<DVector<f64> as Vector<f64>>::new_with_length(0).is_empty());
    assert!(!<DVector<f64> as Vector<f64>>::new_with_length(3).is_empty());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_is_empty_nalgebra_svector() {
    assert!(<SVector<f64, 0> as Vector<f64>>::new_with_length(0).is_empty());
    assert!(!<SVector<f64, 3> as Vector<f64>>::new_with_length(3).is_empty());
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_is_empty_ndarray_array1() {
    assert!(<Array1<f64> as Vector<f64>>::new_with_length(0).is_empty());
    assert!(!<Array1<f64> as Vector<f64>>::new_with_length(3).is_empty());
}
