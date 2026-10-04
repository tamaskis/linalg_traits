use crate::{MulVec, RealField};
use ndarray::{Array1, Array2};

impl<R: RealField> MulVec<R, Array1<R>> for Array2<R> {
    type Output = Array1<R>;

    fn mul_vec(&self, rhs: &Array1<R>) -> Array1<R> {
        self.assert_mul_vec_compatible(rhs);
        self.dot(rhs)
    }
}
