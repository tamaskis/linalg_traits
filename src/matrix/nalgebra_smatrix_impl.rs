use crate::matrix::matrix_trait::Matrix;
use crate::scalar::Scalar;

#[cfg(feature = "with_nalgebra")]
use nalgebra::{SMatrix, SVector};

#[cfg(feature = "with_nalgebra")]
impl<S, const M: usize, const N: usize> Matrix<S> for SMatrix<S, M, N>
where
    S: Scalar,
{
    type VectorM = SVector<S, M>;

    type VectorN = SVector<S, N>;

    fn is_row_major() -> bool {
        false
    }

    fn is_column_major() -> bool {
        true
    }

    fn new_with_shape(rows: usize, cols: usize) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::<S, M, N>::zeros()
    }

    fn shape(&self) -> (usize, usize) {
        (M, N)
    }

    fn from_row_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::from_row_slice(slice)
    }

    fn from_col_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::from_column_slice(slice)
    }

    fn as_slice(&self) -> &[S] {
        Self::as_slice(self)
    }

    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }

    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    fn sub_assign(&mut self, other: &Self) {
        *self -= *other;
    }

    fn mul(&self, scalar: S) -> Self {
        *self * scalar
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= scalar
    }

    fn div(&self, scalar: S) -> Self {
        *self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= scalar
    }
}
