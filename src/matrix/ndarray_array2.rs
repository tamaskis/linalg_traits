use crate::{Matrix, RealField};
use ndarray::{Array1, Array2};
use std::borrow::Cow;

impl<R> Matrix<R> for Array2<R>
where
    R: RealField,
{
    type VectorM = Array1<R>;

    type VectorN = Array1<R>;

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

    fn from_row_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
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

    fn from_col_slice(rows: usize, cols: usize, slice: &[R]) -> Self {
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

    fn as_slice(&self) -> Cow<'_, [R]> {
        match self.as_slice_memory_order() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(self.iter().copied().collect()),
        }
    }

    fn get(&self, index: (usize, usize)) -> Option<&R> {
        let (row, col) = index;
        if row < self.nrows() && col < self.ncols() {
            Some(&self[(row, col)])
        } else {
            None
        }
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
