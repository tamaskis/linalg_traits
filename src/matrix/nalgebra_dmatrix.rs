use crate::{Matrix, RealField};
use nalgebra::{DMatrix, DVector};
use std::borrow::Cow;

impl<R> Matrix<R> for DMatrix<R>
where
    R: RealField,
{
    type VectorM = DVector<R>;

    type VectorN = DVector<R>;

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

    fn from_row_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
        DMatrix::from_row_slice(rows, cols, slice)
    }

    fn from_col_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
        DMatrix::from_column_slice(rows, cols, slice)
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(Self::as_slice(self))
    }

    fn get(&self, index: (usize, usize)) -> Option<&R> {
        DMatrix::get(self, index)
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

    fn mul(&self, scalar: R) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: R) {
        *self *= scalar;
    }

    fn div(&self, scalar: R) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: R) {
        *self /= scalar;
    }
}
