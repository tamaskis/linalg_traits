use crate::{MulMat, RealField};
use ndarray::Array2;

impl<R: RealField> MulMat<R, Array2<R>> for Array2<R> {
    type Output = Array2<R>;

    fn mul_mat(&self, rhs: &Array2<R>) -> Array2<R> {
        self.assert_mul_mat_compatible(rhs);
        self.dot(rhs)
    }
}
