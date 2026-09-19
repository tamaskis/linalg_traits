use crate::{Matrix, RealField, Vector};
use std::borrow::Cow;
use std::iter::Iterator;
use std::ops::{Index, IndexMut};

/// Extremely basic matrix type, written as `Mat<R>`, short for "matrix".
///
/// # Implementation Details
///
/// * The underlying data structure is a [`Vec<R>`].
/// * This matrix implementation is row-major; the elements of the matrix are stored row-by-row
///   in a one-dimensional "flat" data structure (in this case a [`Vec<R>`]).
///
/// # Motivation
///
/// Rust does not have a matrix type in the `std` library, and users of this crate may not want to
/// have dependencies such as [`nalgebra`], [`ndarray`], and/or [`faer`].
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<R>
where
    R: RealField,
{
    data: Vec<R>,
    rows: usize,
    cols: usize,
}

impl<R> Mat<R>
where
    R: RealField,
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
    pub fn iter(&self) -> impl Iterator<Item = &R> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the elements of the matrix.
    ///
    /// # Returns
    ///
    /// An iterator that yields mutable references to the elements of the matrix.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut R> {
        self.data.iter_mut()
    }
}

impl<R> IntoIterator for Mat<R>
where
    R: RealField,
{
    type Item = R;
    type IntoIter = std::vec::IntoIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<R: RealField> Index<(usize, usize)> for Mat<R> {
    type Output = R;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[self.index(row, col)]
    }
}

impl<R: RealField> IndexMut<(usize, usize)> for Mat<R> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let idx = self.index(row, col);
        &mut self.data[idx]
    }
}

impl<R> Matrix<R> for Mat<R>
where
    R: RealField,
{
    type VectorM = Vec<R>;

    type VectorN = Vec<R>;

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
            data: vec![R::_zero(); rows * cols],
            rows,
            cols,
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
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
        Mat {
            data: slice.to_vec(),
            rows,
            cols,
        }
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
        Mat { data, rows, cols }
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(self.data.as_slice())
    }

    fn get(&self, index: (usize, usize)) -> Option<&R> {
        let (row, col) = index;
        let (rows, cols) = self.shape();
        if row < rows && col < cols {
            Some(&self[index])
        } else {
            None
        }
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

    fn mul(&self, scalar: R) -> Self {
        Mat {
            data: self.data.mul(scalar),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn mul_assign(&mut self, scalar: R) {
        self.data.mul_assign(scalar);
    }

    fn div(&self, scalar: R) -> Self {
        Mat {
            data: self.data.div(scalar),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn div_assign(&mut self, scalar: R) {
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
