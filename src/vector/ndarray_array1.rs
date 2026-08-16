use crate::{RealField, Vector};
use ndarray::linalg::Dot;
use ndarray::{Array1, Array2};
use std::borrow::Cow;

impl<R: RealField> Vector<R> for Array1<R> {
    type VectorT<T: RealField> = Array1<T>;

    type DVectorT<T: RealField> = Array1<T>;

    type Vectorf64 = Array1<f64>;

    type DVectorf64 = Array1<f64>;

    type MatrixNxN = Array2<R>;

    type MatrixMxN<const M: usize> = Array2<R>;

    type DMatrixMxN = Array2<R>;

    type DMatrixMxNf64 = Array2<f64>;

    type MatrixNxM<const M: usize> = Array2<R>;

    type DMatrixNxM = Array2<R>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Self {
        Array1::<R>::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn from_slice(slice: &[R]) -> Self {
        Array1::from(slice.to_vec())
    }

    fn as_slice(&self) -> Cow<'_, [R]> {
        match self.as_slice_memory_order() {
            Some(slice) => Cow::from(slice),
            None => panic!("Array1 is not in standard layout for as_slice conversion"),
        }
    }

    fn get(&self, idx: usize) -> Option<&R> {
        match self.as_slice_memory_order() {
            Some(slice) => slice.get(idx),
            None => {
                if idx < self.len() {
                    Some(&self[idx])
                } else {
                    None
                }
            }
        }
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= *b;
        }
    }

    fn mul(&self, scalar: R) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: R) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: R) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: R) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }

    fn dot(&self, other: &Self) -> R {
        Dot::dot(self, other)
    }
}
