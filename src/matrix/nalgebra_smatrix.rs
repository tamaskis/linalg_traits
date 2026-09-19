use crate::{Matrix, RealField};
use nalgebra::{SMatrix, SVector};
use std::borrow::Cow;

impl<R, const M: usize, const N: usize> Matrix<R> for SMatrix<R, M, N>
where
    R: RealField,
{
    type VectorM = SVector<R, M>;

    type VectorN = SVector<R, N>;

    fn is_statically_sized() -> bool {
        true
    }

    fn is_dynamically_sized() -> bool {
        false
    }

    fn is_row_major() -> bool {
        false
    }

    fn is_column_major() -> bool {
        true
    }

    fn new_with_shape(rows: usize, cols: usize) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::<R, M, N>::zeros()
    }

    fn shape(&self) -> (usize, usize) {
        (M, N)
    }

    fn from_row_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::from_row_slice(slice)
    }

    fn from_col_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
        assert_eq!(rows, M, "Row count mismatch.");
        assert_eq!(cols, N, "Column count mismatch.");
        SMatrix::from_column_slice(slice)
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(Self::as_slice(self))
    }

    fn get(&self, index: (usize, usize)) -> Option<&R> {
        SMatrix::get(self, index)
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

    fn mul(&self, scalar: R) -> Self {
        *self * scalar
    }

    fn mul_assign(&mut self, scalar: R) {
        *self *= scalar;
    }

    fn div(&self, scalar: R) -> Self {
        *self / scalar
    }

    fn div_assign(&mut self, scalar: R) {
        *self /= scalar;
    }
}
