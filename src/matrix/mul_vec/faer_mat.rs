use crate::{MulVec, RealField};
use faer::{Col, Mat};

impl<R: RealField> MulVec<R, Col<R>> for Mat<R> {
    type Output = Col<R>;

    fn mul_vec(&self, rhs: &Col<R>) -> Col<R> {
        self.assert_mul_vec_compatible(rhs);
        self * rhs
    }
}
