use linalg_traits::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

#[test]
fn test_mat() {
    assert!(Vec::<f64>::is_dynamically_sized());
    assert!(!Vec::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_dvector() {
    assert!(DVector::<f64>::is_dynamically_sized());
    assert!(!DVector::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "nalgebra")]
fn test_nalgebra_svector() {
    assert!(!SVector::<f64, 3>::is_dynamically_sized());
    assert!(SVector::<f64, 3>::is_statically_sized());
}

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_array1() {
    assert!(Array1::<f64>::is_dynamically_sized());
    assert!(!Array1::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "faer")]
fn test_faer_mat() {
    assert!(FMat::<f64>::is_dynamically_sized());
    assert!(!FMat::<f64>::is_statically_sized());
}
