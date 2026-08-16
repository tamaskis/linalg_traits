use crate::{Mat, RealField, Vector};
use std::borrow::Cow;

impl<R: RealField> Vector<R> for Vec<R> {
    type VectorT<T: RealField> = Vec<T>;

    type DVectorT<T: RealField> = Vec<T>;

    type Vectorf64 = Vec<f64>;

    type DVectorf64 = Vec<f64>;

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

    fn new_with_length(len: usize) -> Vec<R> {
        vec![R::_zero(); len]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[R]) -> Vec<R> {
        slice.to_vec()
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        Cow::from(self.as_slice())
    }

    fn get(&self, idx: usize) -> Option<&R> {
        self.as_slice().get(idx)
    }

    fn add(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a + *b)
            .collect()
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a - *b)
            .collect()
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= *b;
        }
    }

    fn mul(&self, scalar: R) -> Self {
        self.iter().map(|a| *a * scalar).collect()
    }

    fn mul_assign(&mut self, scalar: R) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: R) -> Self {
        self.iter().map(|a| *a / scalar).collect()
    }

    fn div_assign(&mut self, scalar: R) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }

    fn dot(&self, other: &Self) -> R {
        assert_eq!(
            self.len(),
            other.len(),
            "Cannot evaluate the dot product of vectors with different lengths."
        );
        let mut result = R::_zero();
        for i in 0..self.len() {
            result += self[i] * other[i];
        }
        result
    }
}
