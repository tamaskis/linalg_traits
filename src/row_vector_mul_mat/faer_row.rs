use crate::{RealField, RowVectorMulMat};
use faer::{Mat, Row};

impl<R: RealField> RowVectorMulMat<R, Mat<R>> for Row<R> {
    type Output = Row<R>;

    fn mul_mat(&self, rhs: &Mat<R>) -> Row<R> {
        self.assert_mul_mat_compatible(rhs);
        self * rhs
    }
}
