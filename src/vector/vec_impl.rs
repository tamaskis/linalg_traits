use crate::vector::vector_trait::Vector;

impl Vector for Vec<f64> {
    fn new_with_length(len: usize) -> Vec<f64> {
        vec![0.0; len]
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}
