use linalg_traits::Vector;
use numtest::*;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DVector, SVector, Vector1, Vector2, Vector3, Vector4, Vector5, Vector6};

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[test]
#[cfg(feature = "with_nalgebra")]
#[cfg(feature = "with_ndarray")]
fn test_new_with_length() {
    assert_arrays_equal!(Vec::new_with_length(3), [0.0, 0.0, 0.0]);
    assert_arrays_equal!(DVector::new_with_length(3), [0.0, 0.0, 0.0]);
    let v_svector: SVector<f64, 3> = SVector::new_with_length(3);
    assert_arrays_equal!(v_svector, [0.0, 0.0, 0.0]);
    assert_arrays_equal!(Vector1::new_with_length(1), [0.0]);
    assert_arrays_equal!(Vector2::new_with_length(2), [0.0, 0.0]);
    assert_arrays_equal!(Vector3::new_with_length(3), [0.0, 0.0, 0.0]);
    assert_arrays_equal!(Vector4::new_with_length(4), [0.0, 0.0, 0.0, 0.0]);
    assert_arrays_equal!(Vector5::new_with_length(5), [0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_arrays_equal!(Vector6::new_with_length(6), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_arrays_equal!(Array1::new_with_length(3), [0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_svector_error() {
    let _: SVector<f64, 2> = SVector::new_with_length(3);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector1_error() {
    let _ = Vector1::new_with_length(2);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector2_error() {
    let _ = Vector2::new_with_length(3);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector3_error() {
    let _ = Vector3::new_with_length(4);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector4_error() {
    let _ = Vector4::new_with_length(5);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector5_error() {
    let _ = Vector5::new_with_length(6);
}

#[test]
#[should_panic(expected = "Length must match the fixed size of the SVector.")]
#[cfg(feature = "with_nalgebra")]
fn test_new_with_length_vector6_error() {
    let _ = Vector6::new_with_length(4);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_len() {
    assert_eq!(Vec::new_with_length(3).len(), 3);
    assert_eq!(DVector::new_with_length(3).len(), 3);
    let v_svector: SVector<f64, 3> = SVector::new_with_length(3);
    assert_eq!(v_svector.len(), 3);
    assert_eq!(Vector1::new_with_length(1).len(), 1);
    assert_eq!(Vector2::new_with_length(2).len(), 2);
    assert_eq!(Vector3::new_with_length(3).len(), 3);
    assert_eq!(Vector4::new_with_length(4).len(), 4);
    assert_eq!(Vector5::new_with_length(5).len(), 5);
    assert_eq!(Vector6::new_with_length(6).len(), 6);
    assert_eq!(Array1::new_with_length(3).len(), 3);
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_is_empty() {
    // Not empty tests.
    assert!(!Vec::new_with_length(3).is_empty());
    assert!(!DVector::new_with_length(3).is_empty());
    let v_svector: SVector<f64, 3> = SVector::new_with_length(3);
    assert!(!v_svector.is_empty());
    assert!(!Vector1::new_with_length(1).is_empty());
    assert!(!Vector2::new_with_length(2).is_empty());
    assert!(!Vector3::new_with_length(3).is_empty());
    assert!(!Vector4::new_with_length(4).is_empty());
    assert!(!Vector5::new_with_length(5).is_empty());
    assert!(!Vector6::new_with_length(6).is_empty());
    assert!(!Array1::new_with_length(3).is_empty());

    // Empty tests.
    assert!(Vec::new_with_length(0).is_empty());
    assert!(DVector::new_with_length(0).is_empty());
    let v_svector: SVector<f64, 0> = SVector::new_with_length(0);
    assert!(v_svector.is_empty());
    assert!(Array1::new_with_length(0).is_empty());
}
