use linalg_traits::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[test]
fn test_mat() {
    assert!(Vec::<f64>::is_dynamically_sized());
    assert!(!Vec::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dvector() {
    assert!(DVector::<f64>::is_dynamically_sized());
    assert!(!DVector::<f64>::is_statically_sized());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_svector() {
    assert!(!SVector::<f64, 3>::is_dynamically_sized());
    assert!(SVector::<f64, 3>::is_statically_sized());
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array1() {
    assert!(Array1::<f64>::is_dynamically_sized());
    assert!(!Array1::<f64>::is_statically_sized());
}
