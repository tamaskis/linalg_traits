use crate::example::my_scalar::MyScalar;
use crate::{impl_faer_traits_real_field, verify_trait_implemented};

const _: bool = verify_trait_implemented!(MyScalar: faer_traits::ComplexField);
const _: bool = verify_trait_implemented!(MyScalar: faer_traits::RealField);

impl_faer_traits_real_field!(MyScalar);
