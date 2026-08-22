/// Additional requirements to add on top of [`crate::real_field::RealFieldBase`] to make a type
/// compatible with [`faer`] when the `faer` feature is enabled.
///
/// When the `faer` feature is _NOT_ enabled, this trait is simply an empty marker trait.
///
/// # Blanket Implementations
///
/// * When the `faer` feature is disabled, this trait is automatically implemented for any type that
///   implements [`crate::real_field::RealFieldBase`].
/// * When the `faer` feature is enabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`] and [`faer_traits::RealField`].
///     * If a type already implements [`crate::real_field::RealFieldBase`], the
///       [`faer_traits::RealField`] trait can be implemented using [`crate::impl_faer_traits_real_field`].
#[cfg(feature = "faer")]
pub trait RealFieldFaer: faer_traits::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "faer"))]
pub trait RealFieldFaer {}

// Blanket implementations.
#[cfg(feature = "faer")]
impl<T> RealFieldFaer for T where T: crate::real_field::RealFieldBase + faer_traits::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "faer"))]
impl<T> RealFieldFaer for T where T: crate::real_field::RealFieldBase {}

/// Implement the [`faer_traits::RealField`] trait for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`faer_traits::RealField`] trait is being implemented.
#[cfg(feature = "faer")]
#[macro_export]
macro_rules! impl_faer_traits_real_field {
    ($t:ty) => {
        impl faer_traits::ComplexField for $t {
            type Arch = <f64 as faer_traits::ComplexField>::Arch;
            type Unit = f64;
            type Index = <f64 as faer_traits::ComplexField>::Index;
            type Real = $t;

            type SimdCtx<S: faer_traits::pulp::Simd> =
                <f64 as faer_traits::ComplexField>::SimdCtx<S>;
            type SimdMask<S: faer_traits::pulp::Simd> =
                <f64 as faer_traits::ComplexField>::SimdMask<S>;
            type SimdMemMask<S: faer_traits::pulp::Simd> =
                <f64 as faer_traits::ComplexField>::SimdMemMask<S>;
            type SimdVec<S: faer_traits::pulp::Simd> =
                <f64 as faer_traits::ComplexField>::SimdVec<S>;
            type SimdIndex<S: faer_traits::pulp::Simd> =
                <f64 as faer_traits::ComplexField>::SimdIndex<S>;

            const IS_REAL: bool = true;
            const SIMD_CAPABILITIES: faer_traits::SimdCapabilities =
                <f64 as faer_traits::ComplexField>::SIMD_CAPABILITIES;

            #[inline]
            fn zero_impl() -> Self {
                <$t as num_traits::Zero>::zero()
            }

            #[inline]
            fn one_impl() -> Self {
                <$t as num_traits::One>::one()
            }

            #[inline]
            fn nan_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::nan()
            }

            #[inline]
            fn infinity_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::infinity()
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
                <$t as num_traits::Zero>::zero()
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
                <$t as $crate::real_field::RealFieldBase>::recip(*value)
            }

            #[inline]
            fn sqrt_impl(value: &Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::sqrt(*value)
            }

            #[inline]
            fn abs_impl(value: &Self) -> Self::Real {
                <$t as $crate::real_field::RealFieldBase>::abs(*value)
            }

            #[inline]
            fn abs1_impl(value: &Self) -> Self::Real {
                <$t as $crate::real_field::RealFieldBase>::abs(*value) // https://docs.rs/faer-traits/0.24.0/src/faer_traits/lib.rs.html#2161
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
                <$t as $crate::real_field::RealFieldBase>::is_finite(*value)
            }

            #[inline]
            fn simd_ctx<S: faer_traits::pulp::Simd>(simd: S) -> Self::SimdCtx<S> {
                <f64 as faer_traits::ComplexField>::simd_ctx(simd)
            }

            #[inline]
            fn ctx_from_simd<S: faer_traits::pulp::Simd>(ctx: &Self::SimdCtx<S>) -> S {
                <f64 as faer_traits::ComplexField>::ctx_from_simd(ctx)
            }

            #[inline]
            fn simd_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_mask_between(ctx, start, end)
            }

            #[inline]
            fn simd_mem_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMemMask<S> {
                <f64 as faer_traits::ComplexField>::simd_mem_mask_between(ctx, start, end)
            }

            #[inline]
            unsafe fn simd_mask_load_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *const Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                unsafe { <f64 as faer_traits::ComplexField>::simd_mask_load_raw(ctx, mask, ptr) }
            }

            #[inline]
            unsafe fn simd_mask_store_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *mut Self::SimdVec<S>,
                values: Self::SimdVec<S>,
            ) {
                unsafe {
                    <f64 as faer_traits::ComplexField>::simd_mask_store_raw(ctx, mask, ptr, values)
                }
            }

            #[inline]
            fn simd_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self,
            ) -> Self::SimdVec<S> {
                let value_f64: f64 = <$t as Into<f64>>::into(*value);
                <f64 as faer_traits::ComplexField>::simd_splat(ctx, &value_f64)
            }

            #[inline]
            fn simd_splat_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self::Real,
            ) -> Self::SimdVec<S> {
                let value_f64: f64 = <$t as Into<f64>>::into(*value);
                <f64 as faer_traits::ComplexField>::simd_splat_real(ctx, &value_f64)
            }

            #[inline]
            fn simd_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_add(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_sub<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_sub(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_neg<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_neg(ctx, value)
            }

            #[inline]
            fn simd_conj<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_conj(ctx, value)
            }

            #[inline]
            fn simd_abs1<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_abs1(ctx, value)
            }

            #[inline]
            fn simd_abs_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_abs_max(ctx, value)
            }

            #[inline]
            fn simd_mul_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_mul_real(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_mul_pow2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_mul_pow2(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_mul(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_conj_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_conj_mul(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_mul_add(ctx, lhs, rhs, acc)
            }

            #[inline]
            fn simd_conj_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_conj_mul_add(ctx, lhs, rhs, acc)
            }

            #[inline]
            fn simd_abs2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_abs2(ctx, value)
            }

            #[inline]
            fn simd_abs2_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_abs2_add(ctx, value, acc)
            }

            #[inline]
            fn simd_reduce_sum<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                <$t as From<f64>>::from(<f64 as faer_traits::ComplexField>::simd_reduce_sum(
                    ctx, value,
                ))
            }

            #[inline]
            fn simd_reduce_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                <$t as From<f64>>::from(<f64 as faer_traits::ComplexField>::simd_reduce_max(
                    ctx, value,
                ))
            }

            #[inline]
            fn simd_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_equal(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_less_than(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_less_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_less_than_or_equal(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_greater_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_greater_than(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_greater_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_greater_than_or_equal(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                <f64 as faer_traits::ComplexField>::simd_select(ctx, mask, lhs, rhs)
            }

            #[inline]
            fn simd_index_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                <f64 as faer_traits::ComplexField>::simd_index_select(ctx, mask, lhs, rhs)
            }

            #[inline]
            fn simd_index_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::Index,
            ) -> Self::SimdIndex<S> {
                <f64 as faer_traits::ComplexField>::simd_index_splat(ctx, value)
            }

            #[inline]
            fn simd_index_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                <f64 as faer_traits::ComplexField>::simd_index_add(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_index_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_index_less_than(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_and_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_and_mask(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_or_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_or_mask(ctx, lhs, rhs)
            }

            #[inline]
            fn simd_not_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                <f64 as faer_traits::ComplexField>::simd_not_mask(ctx, value)
            }

            #[inline]
            fn simd_first_true_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> usize {
                <f64 as faer_traits::ComplexField>::simd_first_true_mask(ctx, value)
            }
        }

        impl faer_traits::RealField for $t {
            #[inline]
            fn epsilon_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::epsilon()
            }

            #[inline]
            fn nbits_impl() -> usize {
                <$t as $crate::real_field::RealFieldBase>::bits()
            }

            #[inline]
            fn min_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::min_positive()
            }

            #[inline]
            fn max_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::max_positive()
            }

            #[inline]
            fn sqrt_min_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::sqrt_min_positive()
            }

            #[inline]
            fn sqrt_max_positive_impl() -> Self {
                <$t as $crate::real_field::RealFieldBase>::sqrt_max_positive()
            }
        }
    };
}
