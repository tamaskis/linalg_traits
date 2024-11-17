use linalg_traits::{Mat, Matrix};

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, SMatrix};

#[cfg(feature = "with_ndarray")]
use ndarray::Array2;

#[test]
fn test_mat() {
    assert!(Mat::<f64>::is_row_major());
    assert!(!Mat::<f64>::is_column_major());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix() {
    assert!(!DMatrix::<f64>::is_row_major());
    assert!(DMatrix::<f64>::is_column_major());
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_smatrix() {
    assert!(!SMatrix::<f64, 3, 3>::is_row_major());
    assert!(SMatrix::<f64, 3, 3>::is_column_major());
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2() {
    assert!(Array2::<f64>::is_row_major());
    assert!(!Array2::<f64>::is_column_major());
}
