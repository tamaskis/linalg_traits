use crate::matrix::matrix_trait::Matrix;
use crate::scalar::Scalar;

#[cfg(feature = "with_ndarray")]
use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand};

#[cfg(feature = "with_ndarray")]
impl<S> Matrix<S> for Array2<S>
where
    S: Scalar + ScalarOperand + LinalgScalar,
{
    type VectorM = Array1<S>;

    type VectorN = Array1<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn is_row_major() -> bool {
        true
    }

    fn is_column_major() -> bool {
        false
    }

    fn new_with_shape(rows: usize, cols: usize) -> Self {
        Array2::zeros((rows, cols))
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn from_row_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        assert_eq!(
            slice.len(),
            rows * cols,
            "Slice length ({}) not compatible with matrix dimensions ({}x{}).",
            slice.len(),
            rows,
            cols,
        );
        Array2::from_shape_vec((rows, cols), slice.to_vec())
            .expect("Failed to create Array2 from slice")
    }

    fn from_col_slice(rows: usize, cols: usize, slice: &[S]) -> Self {
        assert_eq!(
            slice.len(),
            rows * cols,
            "Slice length ({}) not compatible with matrix dimensions ({}x{}).",
            slice.len(),
            rows,
            cols,
        );
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(slice[row + col * rows]);
            }
        }
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    fn as_slice(&self) -> &[S] {
        Self::as_slice(self).unwrap()
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
        *self *= scalar;
    }

    fn div(&self, scalar: S) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= scalar;
    }
}
