use linalg_traits::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[test]
fn test_len_vec() {
    assert_eq!(<Vec<f64> as Vector>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_len_nalgebra_dvector() {
    assert_eq!(<DVector<f64> as Vector>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_len_nalgebra_svector() {
    assert_eq!(<SVector<f64, 3> as Vector>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_len_ndarray_array1() {
    assert_eq!(<Array1<f64> as Vector>::new_with_length(3).len(), 3);
}
