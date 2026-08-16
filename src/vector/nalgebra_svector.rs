use crate::{RealField, Vector};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use std::borrow::Cow;

impl<const N: usize, R: RealField> Vector<R> for SVector<R, N> {
    type VectorT<T: RealField> = SVector<T, N>;

    type DVectorT<T: RealField> = DVector<T>;

    type Vectorf64 = SVector<f64, N>;

    type DVectorf64 = DVector<f64>;

    type MatrixNxN = SMatrix<R, N, N>;

    type MatrixMxN<const M: usize> = SMatrix<R, M, N>;

    type DMatrixMxN = DMatrix<R>;

    type DMatrixMxNf64 = DMatrix<f64>;

    type MatrixNxM<const M: usize> = SMatrix<R, N, M>;

    type DMatrixNxM = DMatrix<R>;

    fn is_statically_sized() -> bool {
        true
    }

    fn is_dynamically_sized() -> bool {
        false
    }

    fn new_with_length(len: usize) -> Self {
        assert_eq!(len, N, "Length must match the fixed size of the SVector.");
        SVector::from_element(R::zero())
    }

    fn len(&self) -> usize {
        N
    }

    fn is_empty(&self) -> bool {
        false // SVector is never empty because it's fixed size.
    }

    fn from_slice(slice: &[R]) -> Self {
        let mut result = SVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(SVector::as_slice(self))
    }

    fn get(&self, idx: usize) -> Option<&R> {
        SVector::get(self, idx)
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
