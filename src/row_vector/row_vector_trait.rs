use crate::RealField;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

/// Trait defining a generic row vector type.
///
/// A one-dimensional type whose orientation is a row when interacting with matrices.
pub trait RowVector<R: RealField>:
    Clone + Debug + PartialEq + Index<usize, Output = R> + IndexMut<usize>
{
    /// Create a zero-initialized row vector with the given length.
    fn new_with_length(len: usize) -> Self;

    /// Return the number of elements in this row vector.
    fn len(&self) -> usize;

    /// Return whether this row vector has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R: RealField> RowVector<R> for Vec<R> {
    fn new_with_length(len: usize) -> Self {
        vec![R::zero(); len]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "ndarray")]
impl<R: RealField> RowVector<R> for ndarray::Array1<R> {
    fn new_with_length(len: usize) -> Self {
        Self::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nalgebra")]
impl<R: RealField> RowVector<R> for nalgebra::RowDVector<R> {
    fn new_with_length(len: usize) -> Self {
        Self::from_element(len, R::zero())
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nalgebra")]
impl<const N: usize, R: RealField> RowVector<R> for nalgebra::RowSVector<R, N> {
    fn new_with_length(len: usize) -> Self {
        assert_eq!(
            len, N,
            "Length must match the fixed size of the RowSVector."
        );
        Self::from_element(R::zero())
    }

    fn len(&self) -> usize {
        N
    }
}

#[cfg(feature = "faer")]
impl<R: RealField> RowVector<R> for faer::Row<R> {
    fn new_with_length(len: usize) -> Self {
        Self::zeros(len)
    }

    fn len(&self) -> usize {
        self.ncols()
    }
}
