use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MyScalar {
    x: f64,
}

// -----------------------------------------
// Implementing `linalg_traits::RealScalarBase`.
// -----------------------------------------

impl Add for MyScalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        MyScalar { x: self.x + rhs.x }
    }
}

impl AddAssign for MyScalar {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
    }
}

impl AddAssign<f64> for MyScalar {
    fn add_assign(&mut self, rhs: f64) {
        self.x += rhs;
    }
}

impl Add<f64> for MyScalar {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        MyScalar { x: self.x + rhs }
    }
}

impl Sub for MyScalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        MyScalar { x: self.x - rhs.x }
    }
}

impl SubAssign for MyScalar {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
    }
}

impl SubAssign<f64> for MyScalar {
    fn sub_assign(&mut self, rhs: f64) {
        self.x -= rhs;
    }
}

impl Sub<f64> for MyScalar {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self::Output {
        MyScalar { x: self.x - rhs }
    }
}

impl Mul for MyScalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        MyScalar { x: self.x * rhs.x }
    }
}

impl MulAssign for MyScalar {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
    }
}

impl MulAssign<f64> for MyScalar {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
    }
}

impl Mul<f64> for MyScalar {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        MyScalar { x: self.x * rhs }
    }
}

impl Div for MyScalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        MyScalar { x: self.x / rhs.x }
    }
}

impl DivAssign for MyScalar {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
    }
}

impl DivAssign<f64> for MyScalar {
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
    }
}

impl Div<f64> for MyScalar {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        MyScalar { x: self.x / rhs }
    }
}

impl Rem for MyScalar {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        MyScalar { x: self.x % rhs.x }
    }
}

impl RemAssign for MyScalar {
    fn rem_assign(&mut self, rhs: Self) {
        self.x %= rhs.x;
    }
}

impl RemAssign<f64> for MyScalar {
    fn rem_assign(&mut self, rhs: f64) {
        self.x %= rhs;
    }
}

impl Rem<f64> for MyScalar {
    type Output = Self;
    fn rem(self, rhs: f64) -> Self::Output {
        MyScalar { x: self.x % rhs }
    }
}

// impl Float for MyScalar {
//     fn nan() -> Self {
//         MyScalar { x: f64::NAN }
//     }

//     fn infinity() -> Self {
//         MyScalar { x: f64::INFINITY }
//     }

//     fn neg_infinity() -> Self {
//         MyScalar {
//             x: f64::NEG_INFINITY,
//         }
//     }

//     fn is_nan(&self) -> bool {
//         self.x.is_nan()
//     }

//     fn is_infinite(&self) -> bool {
//         self.x.is_infinite()
//     }

//     fn is_finite(&self) -> bool {
//         self.x.is_finite()
//     }

//     fn is_normal(&self) -> bool {
//         self.x.is_normal()
//     }

//     fn classify(&self) -> std::num::FpCategory {
//         self.x.classify()
//     }
// }
