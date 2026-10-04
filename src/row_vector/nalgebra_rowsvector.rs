use crate::RealField;
use crate::row_vector::row_vector_trait::RowVector;

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
