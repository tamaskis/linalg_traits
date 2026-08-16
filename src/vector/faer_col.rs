use crate::{RealField, Vector};
use faer::{Col, Mat, Scale};
use std::borrow::Cow;

impl<R: RealField> Vector<R> for Col<R> {
    type VectorT<T: RealField> = Col<T>;

    type DVectorT<T: RealField> = Col<T>;

    type Vectorf64 = Col<f64>;

    type DVectorf64 = Col<f64>;

    type MatrixNxN = Mat<R>;

    type MatrixMxN<const M: usize> = Mat<R>;

    type DMatrixMxN = Mat<R>;

    type DMatrixMxNf64 = Mat<f64>;

    type MatrixNxM<const M: usize> = Mat<R>;

    type DMatrixNxM = Mat<R>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Self {
        Col::<R>::zeros(len)
    }

    fn len(&self) -> usize {
        self.nrows()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0
    }

    fn from_slice(slice: &[R]) -> Self {
        Col::<R>::from_fn(slice.len(), |i| slice[i])
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::Owned(self.iter().copied().collect())
    }

    fn get(&self, idx: usize) -> Option<&R> {
        if idx < self.len() {
            Some(Col::get(self, idx))
        } else {
            None
        }
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        *self -= other;
    }

    fn mul(&self, scalar: R) -> Self {
        self * Scale(scalar)
    }

    fn mul_assign(&mut self, scalar: R) {
        *self *= Scale(scalar);
    }

    fn div(&self, scalar: R) -> Self {
        self / Scale(scalar)
    }

    fn div_assign(&mut self, scalar: R) {
        *self /= Scale(scalar);
    }

    fn dot(&self, other: &Self) -> R {
        self.assert_same_length(other);
        let mut dot_product = R::zero();
        for i in 0..self.len() {
            dot_product += self[i] * other[i];
        }
        dot_product
    }
}
