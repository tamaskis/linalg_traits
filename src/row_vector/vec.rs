use crate::RealField;
use crate::row_vector::row_vector_trait::RowVector;

impl<R: RealField> RowVector<R> for Vec<R> {
    fn new_with_length(len: usize) -> Self {
        (0..len).map(|_| R::zero()).collect()
    }

    fn len(&self) -> usize {
        self.len()
    }
}
