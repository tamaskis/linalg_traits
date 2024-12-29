use crate::matrix::matrix_trait::Matrix;
use crate::scalar::Scalar;
use crate::Vector;
use std::iter::Iterator;
use std::ops::{Index, IndexMut};

/// Extremely basic matrix type, written as `Mat<S>`, short for "matrix".
///
/// # Implementation Details
///
/// * The underlying data structure is a [`Vec<S>`].
/// * This matrix implementation is row-major; the elements of the matrix are stored row-by-row
///   in a one-dimensional "flat" data structure (in this case a [`Vec<S>`]).
///   `Vec<S>` row by row.
///
/// # Motivation
///
/// Rust does not have a matrix type in the `std` library, and users of this crate may not want to
/// have dependencies such as [`nalgebra`] and/or [`ndarray`].
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<S>
where
    S: Scalar,
{
    data: Vec<S>,
    rows: usize,
    cols: usize,
}

impl<S> Mat<S>
where
    S: Scalar,
{
    /// Helper function to calculate the linear index from row and column indices.
    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        row * self.cols + col
    }

    /// Returns an iterator over the elements of the matrix.
    ///
    /// # Returns
    ///
    /// An iterator that yields references to the elements of the matrix.
    pub fn iter(&self) -> impl Iterator<Item = &S> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the elements of the matrix.
    ///
    /// # Returns
    ///
    /// An iterator that yields mutable references to the elements of the matrix.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.data.iter_mut()
    }
}

impl<S> IntoIterator for Mat<S>
where
    S: Scalar,
{
    type Item = S;
    type IntoIter = std::vec::IntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<S: Scalar> Index<(usize, usize)> for Mat<S> {
    type Output = S;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[self.index(row, col)]
    }
}

impl<S: Scalar> IndexMut<(usize, usize)> for Mat<S> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let idx = self.index(row, col);
        &mut self.data[idx]
    }
}

impl<S> Matrix<S> for Mat<S>
where
    S: Scalar,
{
    type VectorM = Vec<S>;

    type VectorN = Vec<S>;

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
        Mat {
            data: vec![S::zero(); rows * cols],
            rows,
            cols,
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
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
        Mat {
            data: slice.to_vec(),
            rows,
            cols,
        }
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
        Mat { data, rows, cols }
    }

    fn as_slice(&self) -> &[S] {
        self.data.as_slice()
    }

    fn add(&self, other: &Self) -> Self {
        self.assert_same_shape(other);
        Mat {
            data: self.data.add(&other.data),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_shape(other);
        self.data.add_assign(&other.data);
    }

    fn sub(&self, other: &Self) -> Self {
        self.assert_same_shape(other);
        Mat {
            data: self.data.sub(&other.data),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_shape(other);
        self.data.sub_assign(&other.data);
    }

    fn mul(&self, scalar: S) -> Self {
        Mat {
            data: self.data.mul(scalar),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn mul_assign(&mut self, scalar: S) {
        self.data.mul_assign(scalar);
    }

    fn div(&self, scalar: S) -> Self {
        Mat {
            data: self.data.div(scalar),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn div_assign(&mut self, scalar: S) {
        self.data.div_assign(scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing() {
        let mut mat = Mat::<f64>::new_with_shape(2, 2);
        mat[(0, 0)] = 1.0;
        mat[(0, 1)] = 2.0;
        mat[(1, 0)] = 3.0;
        mat[(1, 1)] = 4.0;
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(0, 1)], 2.0);
        assert_eq!(mat[(1, 0)], 3.0);
        assert_eq!(mat[(1, 1)], 4.0);
    }
}
