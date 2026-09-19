/// Implement the [`faer_traits::RealField`] trait for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// [`faer_traits::RealField`] requires [`num_traits::Num`] as a supertrait, so `$t` must already
/// implement [`num_traits::Num`] (e.g. via [`crate::impl_num_traits_num`]) before invoking this
/// macro. In most cases, prefer [`crate::impl_real_field`] instead, which takes care of this for
/// you.
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`faer_traits::RealField`] trait is being implemented.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_faer_traits_real_field {
    ($t:ty) => {
        // Import the re-exported dependency from `__private` so the macro remains hygienic and
        // does not rely on callers having a direct `faer_traits` dependency in scope.
        use $crate::__private::faer_traits;

        impl faer_traits::ComplexField for $t {
            // ------------------------------------------------------------
            // Scalar-only faer implementation.
            //
            // We deliberately do NOT use f64's SIMD representation here.
            // This is important for custom scalar types (for example dual-number based types),
            // where coercing through `f64` would discard additional semantic state.
            // ------------------------------------------------------------

            // Reuse faer's architecture marker type, while keeping this scalar's unit and real
            // components identical to `$t`.
            type Arch = <f64 as faer_traits::ComplexField>::Arch;
            type Unit = $t;
            type Index = usize;
            type Real = $t;

            // Model a scalar-only backend: carry the SIMD context type through, but represent all
            // SIMD vectors/masks as zero-sized placeholders because SIMD code paths are disabled.
            type SimdCtx<S: faer_traits::pulp::Simd> = S;
            type SimdMask<S: faer_traits::pulp::Simd> = ();
            type SimdMemMask<S: faer_traits::pulp::Simd> = ();
            type SimdVec<S: faer_traits::pulp::Simd> = ();
            type SimdIndex<S: faer_traits::pulp::Simd> = ();

            // `$t` is a real scalar (not complex), so imaginary components are always zero.
            const IS_REAL: bool = true;

            // This tells `faer` that this type only supports scalar operations and does not provide
            // SIMD arithmetic. `faer` should therefore use scalar kernels.
            const SIMD_CAPABILITIES: faer_traits::SimdCapabilities =
                faer_traits::SimdCapabilities::Copy;

            #[inline]
            fn zero_impl() -> Self {
                <$t as $crate::__private::num_traits::Zero>::zero()
            }

            #[inline]
            fn one_impl() -> Self {
                <$t as $crate::__private::num_traits::One>::one()
            }

            #[inline]
            fn nan_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_nan()
            }

            #[inline]
            fn infinity_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_infinity()
            }

            #[inline]
            fn from_real_impl(real: &Self::Real) -> Self {
                *real
            }

            #[inline]
            fn from_f64_impl(value: f64) -> Self {
                <$t as From<f64>>::from(value)
            }

            #[inline]
            fn real_part_impl(value: &Self) -> Self::Real {
                *value
            }

            #[inline]
            fn imag_part_impl(_: &Self) -> Self::Real {
                <$t as $crate::__private::num_traits::Zero>::zero()
            }

            #[inline]
            fn copy_impl(value: &Self) -> Self {
                *value
            }

            #[inline]
            fn conj_impl(value: &Self) -> Self {
                *value // https://docs.rs/faer-traits/0.24.0/src/faer_traits/lib.rs.html#2134
            }

            #[inline]
            fn recip_impl(value: &Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_recip(*value)
            }

            #[inline]
            fn sqrt_impl(value: &Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sqrt(*value)
            }

            #[inline]
            fn abs_impl(value: &Self) -> Self::Real {
                <$t as $crate::real_field::RealFieldBase>::_abs(value)
            }

            #[inline]
            fn abs1_impl(value: &Self) -> Self::Real {
                <$t as $crate::real_field::RealFieldBase>::_abs(value) // https://docs.rs/faer-traits/0.24.0/src/faer_traits/lib.rs.html#2161
            }

            #[inline]
            fn abs2_impl(value: &Self) -> Self::Real {
                value * value // https://docs.rs/faer-traits/latest/src/faer_traits/lib.rs.html#2166
            }

            #[inline]
            fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
                lhs * rhs
            }

            #[inline]
            fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
                lhs * rhs // https://docs.rs/faer-traits/0.24.0/src/faer_traits/lib.rs.html#2176
            }

            #[inline]
            fn is_finite_impl(value: &Self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_finite(value)
            }

            // -----------------------------------------------------------------
            // Boilerplate for SIMD-specific methods that will not ever be used.
            // -----------------------------------------------------------------
            // With `SIMD_CAPABILITIES = Copy`, faer will select scalar kernels for this type, so
            // these SIMD entry points should not be reached at runtime.

            #[inline]
            fn simd_ctx<S: faer_traits::pulp::Simd>(simd: S) -> Self::SimdCtx<S> {
                simd
            }

            #[inline]
            fn ctx_from_simd<S: faer_traits::pulp::Simd>(ctx: &Self::SimdCtx<S>) -> S {
                *ctx
            }

            #[inline]
            fn simd_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_mem_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMemMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            unsafe fn simd_mask_load_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *const Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            unsafe fn simd_mask_store_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *mut Self::SimdVec<S>,
                values: Self::SimdVec<S>,
            ) {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_splat_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self::Real,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_sub<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_neg<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_conj<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_abs1<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_abs_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_mul_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_mul_pow2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_conj_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_conj_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_abs2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_abs2_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_reduce_sum<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_reduce_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_less_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_greater_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_greater_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_index_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_index_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::Index,
            ) -> Self::SimdIndex<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_index_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_index_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_and_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_or_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_not_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                unreachable!("SIMD is disabled for this scalar type")
            }

            #[inline]
            fn simd_first_true_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> usize {
                unreachable!("SIMD is disabled for this scalar type")
            }
        }

        impl faer_traits::RealField for $t {
            #[inline]
            fn epsilon_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_epsilon()
            }

            #[inline]
            fn nbits_impl() -> usize {
                <$t as $crate::real_field::RealFieldBase>::_bits()
            }

            #[inline]
            fn min_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_min_positive()
            }

            #[inline]
            fn max_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_max_positive()
            }

            #[inline]
            fn sqrt_min_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sqrt_min_positive()
            }

            #[inline]
            fn sqrt_max_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sqrt_max_positive()
            }
        }
    };
}
