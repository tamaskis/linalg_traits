use crate::my_scalar::MyScalar;
use linalg_traits::{impl_nalgebra_real_field, verify_trait_implemented};

const _: bool = verify_trait_implemented!(MyScalar: nalgebra::ComplexField);
const _: bool = verify_trait_implemented!(MyScalar: nalgebra::RealField);

impl_nalgebra_real_field!(MyScalar);
