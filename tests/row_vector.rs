use linalg_traits::{Mat, Matrix, RealField, RowVector};

fn row_vector_len<R: RealField, V: RowVector<R>>(vector: &V) -> usize {
    RowVector::len(vector)
}

#[test]
fn test_matrix_row_vector() {
    let matrix: Mat<f64> = Mat::new_with_shape(2, 3);
    let row = matrix.new_row_vector();

    assert_eq!(row_vector_len(&row), 2);
}

#[cfg(feature = "nalgebra")]
#[test]
fn test_nalgebra_matrix_row_vector_type() {
    use nalgebra::{DMatrix, RowDVector, RowSVector, SMatrix};

    let dynamic_matrix: DMatrix<f64> = DMatrix::new_with_shape(2, 3);
    let dynamic_row: RowDVector<f64> = dynamic_matrix.new_row_vector();
    assert_eq!(row_vector_len(&dynamic_row), 2);

    let static_matrix: SMatrix<f64, 2, 3> = SMatrix::new_with_shape(2, 3);
    let static_row: RowSVector<f64, 2> = static_matrix.new_row_vector();
    assert_eq!(row_vector_len(&static_row), 2);
}

#[cfg(feature = "faer")]
#[test]
fn test_faer_matrix_row_vector_type() {
    use faer::{Mat, Row};

    let matrix: Mat<f64> = Mat::new_with_shape(2, 3);
    let row: Row<f64> = matrix.new_row_vector();
    assert_eq!(row_vector_len(&row), 2);
}
