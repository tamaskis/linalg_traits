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
