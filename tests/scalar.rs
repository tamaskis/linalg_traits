use linalg_traits::Scalar;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "with_nalgebra")]
use nalgebra::{DMatrix, SMatrix};
use ndarray::Array2;

// Define a custom scalar type for unit testing.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MyType {
    value: f64,
}
impl MyType {
    pub fn new(value: f64) -> MyType {
        MyType { value }
    }
}
impl Add for MyType {
    type Output = MyType;
    fn add(self, other: MyType) -> MyType {
        MyType {
            value: self.value + other.value,
        }
    }
}
impl AddAssign for MyType {
    fn add_assign(&mut self, other: MyType) {
        self.value += other.value;
    }
}
impl Sub for MyType {
    type Output = MyType;
    fn sub(self, other: MyType) -> MyType {
        MyType {
            value: self.value - other.value,
        }
    }
}
impl SubAssign for MyType {
    fn sub_assign(&mut self, other: MyType) {
        self.value -= other.value;
    }
}
impl Mul for MyType {
    type Output = MyType;
    fn mul(self, other: MyType) -> MyType {
        MyType {
            value: self.value * other.value,
        }
    }
}
impl MulAssign for MyType {
    fn mul_assign(&mut self, other: MyType) {
        self.value *= other.value;
    }
}
impl Div for MyType {
    type Output = MyType;
    fn div(self, other: MyType) -> MyType {
        MyType {
            value: self.value / other.value,
        }
    }
}
impl DivAssign for MyType {
    fn div_assign(&mut self, other: MyType) {
        self.value /= other.value;
    }
}
impl Add<f64> for MyType {
    type Output = MyType;
    fn add(self, other: f64) -> MyType {
        MyType {
            value: self.value + other,
        }
    }
}
impl AddAssign<f64> for MyType {
    fn add_assign(&mut self, other: f64) {
        self.value += other;
    }
}
impl Sub<f64> for MyType {
    type Output = MyType;
    fn sub(self, other: f64) -> MyType {
        MyType {
            value: self.value - other,
        }
    }
}
impl SubAssign<f64> for MyType {
    fn sub_assign(&mut self, other: f64) {
        self.value -= other;
    }
}
impl Mul<f64> for MyType {
    type Output = MyType;
    fn mul(self, other: f64) -> MyType {
        MyType {
            value: self.value * other,
        }
    }
}
impl MulAssign<f64> for MyType {
    fn mul_assign(&mut self, other: f64) {
        self.value *= other;
    }
}
impl Div<f64> for MyType {
    type Output = MyType;
    fn div(self, other: f64) -> MyType {
        MyType {
            value: self.value / other,
        }
    }
}
impl DivAssign<f64> for MyType {
    fn div_assign(&mut self, other: f64) {
        self.value /= other;
    }
}
impl Zero for MyType {
    fn zero() -> Self {
        MyType { value: 0.0 }
    }
    fn is_zero(&self) -> bool {
        self.value == 0.0
    }
}

// Testing utility functions.
fn check_add_and_add_assign_self<T: Scalar>(mut x: T, y: T, result: T) {
    assert_eq!(x + y, result);
    x += y;
    assert_eq!(x, result);
}
fn check_sub_and_sub_assign_self<T: Scalar>(mut x: T, y: T, result: T) {
    assert_eq!(x - y, result);
    x -= y;
    assert_eq!(x, result);
}
fn check_mul_and_mul_assign_self<T: Scalar>(mut x: T, y: T, result: T) {
    assert_eq!(x * y, result);
    x *= y;
    assert_eq!(x, result);
}
fn check_div_and_div_assign_self<T: Scalar>(mut x: T, y: T, result: T) {
    assert_eq!(x / y, result);
    x /= y;
    assert_eq!(x, result);
}
fn check_add_and_add_assign_f64<T: Scalar>(mut x: T, y: f64, result: T) {
    assert_eq!(x + y, result);
    x += y;
    assert_eq!(x, result);
}
fn check_sub_and_sub_assign_f64<T: Scalar>(mut x: T, y: f64, result: T) {
    assert_eq!(x - y, result);
    x -= y;
    assert_eq!(x, result);
}
fn check_mul_and_mul_assign_f64<T: Scalar>(mut x: T, y: f64, result: T) {
    assert_eq!(x * y, result);
    x *= y;
    assert_eq!(x, result);
}
fn check_div_and_div_assign_f64<T: Scalar>(mut x: T, y: f64, result: T) {
    assert_eq!(x / y, result);
    x /= y;
    assert_eq!(x, result);
}
#[allow(clippy::clone_on_copy)]
fn check_zero<T: Scalar>(expected: T) {
    assert_eq!(T::zero(), expected);
}
fn check_copy<T: Scalar>(x: T) {
    let y = x;
    assert_eq!(x, y);
}
#[allow(clippy::clone_on_copy)]
fn check_clone<T: Scalar>(x: T) {
    let y = x.clone();
    assert_eq!(x, y);
}
#[allow(clippy::eq_op)]
fn check_partial_eq_partial_ord<T: Scalar>(lower: T, middle: T, upper: T) {
    assert!(lower < middle);
    assert!(lower <= middle);
    assert!(middle <= middle);
    assert!(middle == middle);
    assert!(middle >= middle);
    assert!(upper >= middle);
    assert!(upper > middle);
}
#[allow(clippy::clone_on_copy)]
fn check_debug<T: Scalar>(x: T, expected: String) {
    let debug_str = format!("{:?}", x);
    assert_eq!(debug_str, expected);
}

#[test]
fn test_custom_type() {
    check_add_and_add_assign_self(MyType::new(1.0), MyType::new(2.0), MyType::new(3.0));
    check_sub_and_sub_assign_self(MyType::new(1.0), MyType::new(2.0), MyType::new(-1.0));
    check_mul_and_mul_assign_self(MyType::new(3.0), MyType::new(-4.0), MyType::new(-12.0));
    check_div_and_div_assign_self(MyType::new(3.0), MyType::new(-4.0), MyType::new(-0.75));
    check_add_and_add_assign_f64(MyType::new(1.0), 2.0, MyType::new(3.0));
    check_sub_and_sub_assign_f64(MyType::new(1.0), 2.0, MyType::new(-1.0));
    check_mul_and_mul_assign_f64(MyType::new(3.0), -4.0, MyType::new(-12.0));
    check_div_and_div_assign_f64(MyType::new(3.0), -4.0, MyType::new(-0.75));
    check_zero(MyType::new(0.0));
    check_copy(MyType::new(3.0));
    check_clone(MyType::new(3.0));
    check_partial_eq_partial_ord(MyType::new(1.0), MyType::new(2.0), MyType::new(3.0));
    check_debug(MyType::new(1.0), String::from("MyType { value: 1.0 }"));
}

#[test]
fn test_f64() {
    check_add_and_add_assign_self(1.0, 2.0, 3.0);
    check_sub_and_sub_assign_self(1.0, 2.0, -1.0);
    check_mul_and_mul_assign_self(3.0, -4.0, -12.0);
    check_div_and_div_assign_self(3.0, -4.0, -0.75);
    check_add_and_add_assign_f64(1.0, 2.0, 3.0);
    check_sub_and_sub_assign_f64(1.0, 2.0, -1.0);
    check_mul_and_mul_assign_f64(3.0, -4.0, -12.0);
    check_div_and_div_assign_f64(3.0, -4.0, -0.75);
    check_zero(0.0);
    check_copy(3.0);
    check_clone(3.0);
    check_partial_eq_partial_ord(0.1, 0.2, 0.3);
    check_debug(1.0, String::from("1.0"));
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_dmatrix() {
    let mat = DMatrix::<MyType>::from_row_slice(
        2,
        2,
        &[
            MyType::new(1.0),
            MyType::new(2.0),
            MyType::new(3.0),
            MyType::new(4.0),
        ],
    );
    assert_eq!(mat[(0, 0)], MyType::new(1.0));
    assert_eq!(mat[(0, 1)], MyType::new(2.0));
    assert_eq!(mat[(1, 0)], MyType::new(3.0));
    assert_eq!(mat[(1, 1)], MyType::new(4.0));
}

#[test]
#[cfg(feature = "with_nalgebra")]
fn test_nalgebra_smatrix() {
    let mat = SMatrix::<MyType, 2, 2>::from_row_slice(&[
        MyType::new(1.0),
        MyType::new(2.0),
        MyType::new(3.0),
        MyType::new(4.0),
    ]);
    assert_eq!(mat[(0, 0)], MyType::new(1.0));
    assert_eq!(mat[(0, 1)], MyType::new(2.0));
    assert_eq!(mat[(1, 0)], MyType::new(3.0));
    assert_eq!(mat[(1, 1)], MyType::new(4.0));
}

#[test]
#[cfg(feature = "with_ndarray")]
fn test_ndarray_array2() {
    let mat = Array2::<MyType>::from_shape_vec(
        (2, 2),
        vec![
            MyType::new(1.0),
            MyType::new(2.0),
            MyType::new(3.0),
            MyType::new(4.0),
        ],
    )
    .unwrap();
    assert_eq!(mat[(0, 0)], MyType::new(1.0));
    assert_eq!(mat[(0, 1)], MyType::new(2.0));
    assert_eq!(mat[(1, 0)], MyType::new(3.0));
    assert_eq!(mat[(1, 1)], MyType::new(4.0));
}
