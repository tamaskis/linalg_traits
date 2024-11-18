use crate::matrix::matrix_trait::Matrix;
use crate::scalar::Scalar;

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "nalgebra")]
impl<S> Matrix<S> for DMatrix<S>
where
    S: Scalar,
{
    type VectorM = DVector<S>;

    type VectorN = DVector<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn is_row_major() -> bool {
        false
    }

    fn is_column_major() -> bool {
        true
    }

    fn new_with_shape(rows: usize, cols: usize) -> Self {
        DMatrix::zeros(rows, cols)
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn from_row_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        DMatrix::from_row_slice(rows, cols, slice)
    }

    fn from_col_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        DMatrix::from_column_slice(rows, cols, slice)
    }

    fn as_slice(&self) -> &[S] {
        Self::as_slice(self)
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        *self -= other;
    }

    fn mul(&self, scalar: S) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= scalar
    }

    fn div(&self, scalar: S) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= scalar
    }
}
