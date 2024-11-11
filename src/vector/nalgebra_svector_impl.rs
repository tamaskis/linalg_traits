use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::SVector;

#[cfg(feature = "with_nalgebra")]
// Implementing the Vector trait for a fixed-size vector (SVector)
impl<const N: usize> Vector for SVector<f64, N> {
    fn new_with_length(len: usize) -> Self {
        assert_eq!(len, N, "Length must match the fixed size of the SVector.");
        SVector::from_element(0.0)
    }
    fn len(&self) -> usize {
        N
    }
    fn is_empty(&self) -> bool {
        false // SVector is never empty because it's fixed size.
    }
}
