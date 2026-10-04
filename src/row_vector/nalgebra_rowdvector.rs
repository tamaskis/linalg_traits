use crate::RealField;
use crate::row_vector::row_vector_trait::RowVector;

impl<R: RealField> RowVector<R> for nalgebra::RowDVector<R> {
    fn new_with_length(len: usize) -> Self {
        Self::from_element(len, R::zero())
    }

    fn len(&self) -> usize {
        self.len()
    }
}
