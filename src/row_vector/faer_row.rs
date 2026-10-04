use crate::RealField;
use crate::row_vector::row_vector_trait::RowVector;

impl<R: RealField> RowVector<R> for faer::Row<R> {
    fn new_with_length(len: usize) -> Self {
        Self::zeros(len)
    }

    fn len(&self) -> usize {
        self.ncols()
    }
}
