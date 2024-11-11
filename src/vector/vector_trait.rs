/// Trait defining common vector methods and operations.
///
/// # Note
///
/// In addition to the methods defined by this trait, this trait also forces that the implementor
/// also support indexing ([`std::ops::Index`]) and mutable indexing ([`std::ops::IndexMut`]).
pub trait Vector:
    std::ops::Index<usize, Output = f64> + std::ops::IndexMut<usize, Output = f64> + Clone
{
    /// Create a vector with the specified length, with each element set to 0.
    ///
    /// # Arguments
    ///
    /// * `len` - Desired length of the vector.
    ///
    /// # Returns
    ///
    /// Vector with the specified length, with each element set to 0.
    fn new_with_length(len: usize) -> Self;

    /// Get the length of the vector.
    ///
    /// # Returns
    ///
    /// Length of the vector.
    fn len(&self) -> usize;

    /// Determine if the vector is empty.
    ///
    /// # Returns
    ///
    /// `true` if the vector is empty, `false` if it is not empty.
    fn is_empty(&self) -> bool;
}
