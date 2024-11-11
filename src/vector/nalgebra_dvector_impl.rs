use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::DVector;

#[cfg(feature = "with_nalgebra")]
impl Vector for DVector<f64> {
    fn new_with_length(len: usize) -> DVector<f64> {
        DVector::from_element(len, 0.0)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}
