use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::SVector;

#[cfg(feature = "with_nalgebra")]
impl<const N: usize> Vector for SVector<f64, N> {
    fn new_with_length(len: usize) -> Self {
        assert_eq!(len, N, "Length must match the fixed size of the SVector.");
        SVector::from_element(0.0)
    }

    fn len(&self) -> usize {
        N
    }

    fn is_empty(&self) -> bool {
        false // SVector is never empty because it's fixed size.
    }

    fn from_slice(slice: &[f64]) -> Self {
        let mut result = SVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> &[f64] {
        self.as_slice()
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

    fn mul(&self, scalar: f64) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: f64) {
        *self *= scalar;
    }

    fn div(&self, scalar: f64) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: f64) {
        *self /= scalar;
    }
}
