use crate::{RealField, Vector};
use nalgebra::{DMatrix, DVector};
use std::borrow::Cow;

impl<R: RealField> Vector<R> for DVector<R> {
    type VectorT<T: RealField> = DVector<T>;

    type DVectorT<T: RealField> = DVector<T>;

    type Vectorf64 = DVector<f64>;

    type DVectorf64 = DVector<f64>;

    type MatrixNxN = DMatrix<R>;

    type MatrixMxN<const M: usize> = DMatrix<R>;

    type DMatrixMxN = DMatrix<R>;

    type DMatrixMxNf64 = DMatrix<f64>;

    type MatrixNxM<const M: usize> = DMatrix<R>;

    type DMatrixNxM = DMatrix<R>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> DVector<R> {
        DVector::from_element(len, R::zero())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[R]) -> Self {
        let mut result = DVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(DVector::as_slice(self))
    }

    fn get(&self, idx: usize) -> Option<&R> {
        DVector::get(self, idx)
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
        self * scalar
    }

    fn mul_assign(&mut self, scalar: R) {
        *self *= scalar;
    }

    fn div(&self, scalar: R) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: R) {
        *self /= scalar;
    }

    fn dot(&self, other: &Self) -> R {
        self.dot(other)
    }
}
