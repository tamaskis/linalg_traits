use crate::{Scalar, Vector};
use faer::{Col, Mat, Scale};
use faer_traits::RealField;
use std::borrow::Cow;

impl<S: Scalar + RealField> Vector<S> for Col<S> {
    // Cannot apply RealField trait bound on T because it would be more restrictive than the trait
    // definition.
    type VectorT<T: Scalar> = Vec<T>;

    // Cannot apply RealField trait bound on T because it would be more restrictive than the trait
    // definition.
    type DVectorT<T: Scalar> = Vec<T>;

    type Vectorf64 = Col<f64>;

    type DVectorf64 = Col<f64>;

    type MatrixNxN = Mat<S>;

    type MatrixMxN<const M: usize> = Mat<S>;

    type DMatrixMxN = Mat<S>;

    type DMatrixMxNf64 = Mat<f64>;

    type MatrixNxM<const M: usize> = Mat<S>;

    type DMatrixNxM = Mat<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Self {
        Col::<S>::zeros(len)
    }

    fn len(&self) -> usize {
        self.nrows()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0
    }

    fn from_slice(slice: &[S]) -> Self {
        Col::<S>::from_fn(slice.len(), |i| slice[i])
    }

    fn as_slice(&self) -> Cow<'_, [S]> {
        Cow::Owned(self.iter().copied().collect())
    }

    fn get(&self, idx: usize) -> Option<&S> {
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

    fn mul(&self, scalar: S) -> Self {
        self * Scale(scalar)
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= Scale(scalar);
    }

    fn div(&self, scalar: S) -> Self {
        self / Scale(scalar)
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= Scale(scalar);
    }

    fn dot(&self, other: &Self) -> S {
        self.assert_same_length(other);
        let mut dot_product = S::zero();
        for i in 0..self.len() {
            dot_product += self[i] * other[i];
        }
        dot_product
    }
}
