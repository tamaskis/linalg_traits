//! Implements [`linalg_traits::RealField`] for [`crate::my_real::MyReal`].

use crate::my_real::MyReal;
use linalg_traits::{RealField, verify_trait_implemented};

const _: bool = verify_trait_implemented!(MyReal: RealField);

linalg_traits::impl_real_field!(MyReal);
