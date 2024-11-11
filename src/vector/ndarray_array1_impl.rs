use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[cfg(feature = "with_ndarray")]
impl Vector for Array1<f64> {
    fn new_with_length(len: usize) -> Self {
        Array1::<f64>::zeros(len)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
