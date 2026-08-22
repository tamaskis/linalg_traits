use crate::my_scalar::MyScalar;
use linalg_traits::real_field_operations::RealFieldOperations;
use linalg_traits::verify_trait_implemented;

const _: bool = verify_trait_implemented!(MyScalar: RealFieldOperations);

// ------
// Tests.
// ------

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field_operations::assert_real_field_operations;

    #[test]
    fn test_real_field_operations() {
        assert_real_field_operations(
            MyScalar::new(7.0),
            MyScalar::new(2.0),
            MyScalar::new(-7.0),
            MyScalar::new(9.0),
            MyScalar::new(5.0),
            MyScalar::new(14.0),
            MyScalar::new(3.5),
            MyScalar::new(1.0),
            false,
            false,
            false,
            true,
            true,
        );

        // `for<'a> &'a T: Neg` isn't part of the generic helper's bounds (it triggers an
        // unbounded trait-solver cycle through `nalgebra`/`ndarray`'s own `Neg` impls when `T` is
        // abstract), so borrowed negation is verified directly on the concrete type instead.
        assert_eq!(-&MyScalar::new(7.0), MyScalar::new(-7.0));
    }
}
