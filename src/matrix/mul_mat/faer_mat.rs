use crate::{MulMat, RealField};
use faer::Mat;

impl<R: RealField> MulMat<R, Mat<R>> for Mat<R> {
    type Output = Mat<R>;

    fn mul_mat(&self, rhs: &Mat<R>) -> Mat<R> {
        self.assert_mul_mat_compatible(rhs);
        self * rhs
    }
}
