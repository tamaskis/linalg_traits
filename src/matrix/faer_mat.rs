use crate::matrix::matrix_trait::Matrix;
use crate::scalar::Scalar;
use std::borrow::Cow;

#[cfg(feature = "faer")]
use faer::{Mat, Scale};

#[cfg(feature = "faer-traits")]
use faer_traits::RealField;

#[cfg(feature = "faer")]
#[cfg(feature = "faer-traits")]
impl<S> Matrix<S> for Mat<S>
where
    S: Scalar + RealField,
{
    type VectorM = Mat<S>;

    type VectorN = Mat<S>;

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
        Mat::<S>::zeros(rows, cols)
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
        Mat::<S>::from_fn(rows, cols, |i, j| slice[i * cols + j])
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
        Mat::<S>::from_fn(rows, cols, |i, j| slice[i + j * rows])
    }

    fn as_slice<'a>(&'a self) -> Cow<'a, [S]> {
        let mut slice_vec = Vec::<S>::with_capacity(self.nrows() * self.ncols());
        for i in 0..self.ncols() {
            slice_vec.extend_from_slice(self.col_as_slice(i));
        }
        Cow::from(slice_vec)
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
        self * Scale(scalar)
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= Scale(scalar)
    }

    fn div(&self, scalar: S) -> Self {
        self / Scale(scalar)
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= Scale(scalar)
    }
}
