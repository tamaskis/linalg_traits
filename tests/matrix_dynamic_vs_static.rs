use linalg_traits::{Mat, Matrix};

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[test]
fn test_mat() {
    assert!(Mat::<f64>::is_dynamically_sized());
    assert!(!Mat::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dmatrix() {
    assert!(DMatrix::<f64>::is_dynamically_sized());
    assert!(!DMatrix::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_smatrix() {
    assert!(!SMatrix::<f64, 3, 3>::is_dynamically_sized());
    assert!(SMatrix::<f64, 3, 3>::is_statically_sized());
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array2() {
    assert!(Array2::<f64>::is_dynamically_sized());
    assert!(!Array2::<f64>::is_statically_sized());
}
