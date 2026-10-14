use crate::{RealField, RowVectorMulMat};
use ndarray::{Array1, Array2};

impl<R: RealField> RowVectorMulMat<R, Array2<R>> for Array1<R> {
    type Output = Array1<R>;

    fn mul_mat(&self, rhs: &Array2<R>) -> Array1<R> {
        self.assert_mul_mat_compatible(rhs);
        self.dot(rhs)
    }
}
