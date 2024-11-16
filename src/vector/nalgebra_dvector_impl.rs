use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::DVector;

#[cfg(feature = "with_nalgebra")]
impl<S: Scalar> Vector<S> for DVector<S> {
    fn new_with_length(len: usize) -> DVector<S> {
        DVector::from_element(len, S::zero())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[S]) -> Self {
        let mut result = DVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> &[S] {
        self.as_slice()
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

    fn mul(&self, scalar: S) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: S) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }

    fn dot(&self, other: &Self) -> S {
        self.dot(other)
    }
}
