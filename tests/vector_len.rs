use linalg_traits::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

#[test]
fn test_len_vec() {
    assert_eq!(<Vec<f64> as Vector<f64>>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_len_nalgebra_dvector() {
    assert_eq!(<DVector<f64> as Vector<f64>>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_len_nalgebra_svector() {
    assert_eq!(
        <SVector<f64, 3> as Vector<f64>>::new_with_length(3).len(),
        3
    );
}

#[test]
#[cfg(feature = "ndarray")]
fn test_len_ndarray_array1() {
    assert_eq!(<Array1<f64> as Vector<f64>>::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "faer")]
fn test_len_faer_mat() {
    assert_eq!(FMat::<f64>::new_with_length(3).len(), 3);
}
