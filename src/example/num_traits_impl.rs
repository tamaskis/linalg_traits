use num_traits::ToPrimitive;

use crate::example::my_scalar::MyScalar;
use crate::verify_trait_implemented;

const _: bool = verify_trait_implemented!(MyScalar: num_traits::Num);

impl num_traits::Zero for MyScalar {
    #[inline]
    fn zero() -> Self {
        Self::new(0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl num_traits::One for MyScalar {
    #[inline]
    fn one() -> Self {
        Self::new(1.0)
    }
}

impl num_traits::Num for MyScalar {
    type FromStrRadixErr = core::num::ParseFloatError;

    #[inline]
    fn from_str_radix(value: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix == 10 {
            value.parse::<f64>().map(MyScalar::from)
        } else {
            "invalid radix".parse::<f64>().map(MyScalar::from)
        }
    }
}

impl num_traits::Signed for MyScalar {
    #[inline]
    fn abs(&self) -> Self {
        <MyScalar as crate::real_field::RealFieldBase>::abs(*self)
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            <MyScalar as num_traits::Zero>::zero()
        } else {
            *self - *other
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        if <MyScalar as crate::real_field::RealFieldBase>::is_nan(*self) {
            *self
        } else if <MyScalar as crate::real_field::RealFieldBase>::is_sign_negative(*self) {
            -<MyScalar as num_traits::One>::one()
        } else if <MyScalar as num_traits::Zero>::is_zero(self) {
            <MyScalar as num_traits::Zero>::zero()
        } else {
            <MyScalar as num_traits::One>::one()
        }
    }

    #[inline]
    fn is_positive(&self) -> bool {
        !<MyScalar as num_traits::Zero>::is_zero(self)
            && <MyScalar as crate::real_field::RealFieldBase>::is_sign_positive(*self)
    }

    #[inline]
    fn is_negative(&self) -> bool {
        <MyScalar as crate::real_field::RealFieldBase>::is_sign_negative(*self)
    }
}

impl num_traits::FromPrimitive for MyScalar {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        n.to_f64().map(MyScalar::from)
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        n.to_f64().map(MyScalar::from)
    }
}
