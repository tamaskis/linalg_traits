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
///
/// TODO simplify this impl as much as possible.
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

    (
        real = $real:ty,
        complex = $complex:ty,

        arch = $arch:ty,
        unit = $unit:ty,
        index = $index:ty,

        simd_ctx = $simd_ctx:ty,
        simd_mask = $simd_mask:ty,
        simd_mem_mask = $simd_mem_mask:ty,
        simd_vec = $simd_vec:ty,
        simd_index = $simd_index:ty,

        is_real = $is_real:expr,
        simd_capabilities = $simd_capabilities:expr,
        simd_abs_split_real_imag = $simd_abs_split_real_imag:expr,

        // Scalar operations
        zero = $zero:expr,
        one = $one:expr,
        nan = $nan:expr,
        infinity = $infinity:expr,
        from_real = $from_real:expr,
        from_f64 = $from_f64:expr,
        real_part = $real_part:expr,
        imag_part = $imag_part:expr,
        copy = $copy:expr,
        conj = $conj:expr,
        recip = $recip:expr,
        sqrt = $sqrt:expr,
        abs = $abs:expr,
        abs1 = $abs1:expr,
        abs2 = $abs2:expr,
        mul_real = $mul_real:expr,
        mul_pow2 = $mul_pow2:expr,
        is_finite = $is_finite:expr,

        // SIMD operations
        simd_ctx_fn = $simd_ctx_fn:expr,
        ctx_from_simd = $ctx_from_simd:expr,
        simd_mask_between = $simd_mask_between:expr,
        simd_mem_mask_between = $simd_mem_mask_between:expr,
        simd_mask_load_raw = $simd_mask_load_raw:expr,
        simd_mask_store_raw = $simd_mask_store_raw:expr,
        simd_splat = $simd_splat:expr,
        simd_splat_real = $simd_splat_real:expr,
        simd_add = $simd_add:expr,
        simd_sub = $simd_sub:expr,
        simd_neg = $simd_neg:expr,
        simd_conj = $simd_conj:expr,
        simd_abs1 = $simd_abs1:expr,
        simd_abs_max = $simd_abs_max:expr,
        simd_mul_real = $simd_mul_real:expr,
        simd_mul_pow2 = $simd_mul_pow2:expr,
        simd_mul = $simd_mul:expr,
        simd_conj_mul = $simd_conj_mul:expr,
        simd_mul_add = $simd_mul_add:expr,
        simd_conj_mul_add = $simd_conj_mul_add:expr,
        simd_abs2 = $simd_abs2:expr,
        simd_abs2_add = $simd_abs2_add:expr,
        simd_reduce_sum = $simd_reduce_sum:expr,
        simd_reduce_max = $simd_reduce_max:expr,
        simd_equal = $simd_equal:expr,
        simd_less_than = $simd_less_than:expr,
        simd_less_than_or_equal = $simd_less_than_or_equal:expr,
        simd_greater_than = $simd_greater_than:expr,
        simd_greater_than_or_equal = $simd_greater_than_or_equal:expr,
        simd_select = $simd_select:expr,
        simd_index_select = $simd_index_select:expr,
        simd_index_splat = $simd_index_splat:expr,
        simd_index_add = $simd_index_add:expr,
        simd_index_less_than = $simd_index_less_than:expr,
        simd_and_mask = $simd_and_mask:expr,
        simd_or_mask = $simd_or_mask:expr,
        simd_not_mask = $simd_not_mask:expr,
        simd_first_true_mask = $simd_first_true_mask:expr,

        // RealField
        epsilon = $epsilon:expr,
        nbits = $nbits:expr,
        min_positive = $min_positive:expr,
        max_positive = $max_positive:expr,
        sqrt_min_positive = $sqrt_min_positive:expr,
        sqrt_max_positive = $sqrt_max_positive:expr,
    ) => {
        impl faer_traits::ComplexField for $complex {
            type Arch = $arch;
            type Unit = $unit;
            type Index = $index;
            type Real = $real;

            type SimdCtx<S: faer_traits::pulp::Simd> = $simd_ctx;

            type SimdMask<S: faer_traits::pulp::Simd> = $simd_mask;

            type SimdMemMask<S: faer_traits::pulp::Simd> = $simd_mem_mask;

            type SimdVec<S: faer_traits::pulp::Simd> = $simd_vec;

            type SimdIndex<S: faer_traits::pulp::Simd> = $simd_index;

            const IS_REAL: bool = $is_real;

            const SIMD_CAPABILITIES: faer_traits::SimdCapabilities = $simd_capabilities;

            const SIMD_ABS_SPLIT_REAL_IMAG: bool = $simd_abs_split_real_imag;

            #[inline]
            fn zero_impl() -> Self {
                $zero
            }

            #[inline]
            fn one_impl() -> Self {
                $one
            }

            #[inline]
            fn nan_impl() -> Self {
                $nan
            }

            #[inline]
            fn infinity_impl() -> Self {
                $infinity
            }

            #[inline]
            fn from_real_impl(real: &Self::Real) -> Self {
                $from_real
            }

            #[inline]
            fn from_f64_impl(value: f64) -> Self {
                $from_f64
            }

            #[inline]
            fn real_part_impl(value: &Self) -> Self::Real {
                $real_part
            }

            #[inline]
            fn imag_part_impl(value: &Self) -> Self::Real {
                $imag_part
            }

            #[inline]
            fn copy_impl(value: &Self) -> Self {
                $copy
            }

            #[inline]
            fn conj_impl(value: &Self) -> Self {
                $conj
            }

            #[inline]
            fn recip_impl(value: &Self) -> Self {
                $recip
            }

            #[inline]
            fn sqrt_impl(value: &Self) -> Self {
                $sqrt
            }

            #[inline]
            fn abs_impl(value: &Self) -> Self::Real {
                $abs
            }

            #[inline]
            fn abs1_impl(value: &Self) -> Self::Real {
                $abs1
            }

            #[inline]
            fn abs2_impl(value: &Self) -> Self::Real {
                $abs2
            }

            #[inline]
            fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
                $mul_real
            }

            #[inline]
            fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
                $mul_pow2
            }

            #[inline]
            fn is_finite_impl(value: &Self) -> bool {
                $is_finite
            }

            // -------------------------------------------------------------
            // SIMD
            // -------------------------------------------------------------

            #[inline]
            fn simd_ctx<S: faer_traits::pulp::Simd>(simd: S) -> Self::SimdCtx<S> {
                $simd_ctx_fn
            }

            #[inline]
            fn ctx_from_simd<S: faer_traits::pulp::Simd>(ctx: &Self::SimdCtx<S>) -> S {
                $ctx_from_simd
            }

            #[inline]
            fn simd_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMask<S> {
                $simd_mask_between
            }

            #[inline]
            fn simd_mem_mask_between<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                start: Self::Index,
                end: Self::Index,
            ) -> Self::SimdMemMask<S> {
                $simd_mem_mask_between
            }

            #[inline]
            unsafe fn simd_mask_load_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *const Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_mask_load_raw
            }

            #[inline]
            unsafe fn simd_mask_store_raw<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMemMask<S>,
                ptr: *mut Self::SimdVec<S>,
                values: Self::SimdVec<S>,
            ) {
                $simd_mask_store_raw
            }

            #[inline]
            fn simd_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self,
            ) -> Self::SimdVec<S> {
                $simd_splat
            }

            #[inline]
            fn simd_splat_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: &Self::Real,
            ) -> Self::SimdVec<S> {
                $simd_splat_real
            }

            #[inline]
            fn simd_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_add
            }

            #[inline]
            fn simd_sub<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_sub
            }

            #[inline]
            fn simd_neg<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_neg
            }

            #[inline]
            fn simd_conj<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_conj
            }

            #[inline]
            fn simd_abs1<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_abs1
            }

            #[inline]
            fn simd_abs_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_abs_max
            }

            #[inline]
            fn simd_mul_real<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_mul_real
            }

            #[inline]
            fn simd_mul_pow2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_mul_pow2
            }

            #[inline]
            fn simd_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_mul
            }

            #[inline]
            fn simd_conj_mul<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_conj_mul
            }

            #[inline]
            fn simd_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_mul_add
            }

            #[inline]
            fn simd_conj_mul_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_conj_mul_add
            }

            #[inline]
            fn simd_abs2<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_abs2
            }

            #[inline]
            fn simd_abs2_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
                acc: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_abs2_add
            }

            #[inline]
            fn simd_reduce_sum<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                $simd_reduce_sum
            }

            #[inline]
            fn simd_reduce_max<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdVec<S>,
            ) -> Self {
                $simd_reduce_max
            }

            #[inline]
            fn simd_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                $simd_equal
            }

            #[inline]
            fn simd_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                $simd_less_than
            }

            #[inline]
            fn simd_less_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                $simd_less_than_or_equal
            }

            #[inline]
            fn simd_greater_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                $simd_greater_than
            }

            #[inline]
            fn simd_greater_than_or_equal<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdMask<S> {
                $simd_greater_than_or_equal
            }

            #[inline]
            fn simd_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdVec<S>,
                rhs: Self::SimdVec<S>,
            ) -> Self::SimdVec<S> {
                $simd_select
            }

            #[inline]
            fn simd_index_select<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                mask: Self::SimdMask<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                $simd_index_select
            }

            #[inline]
            fn simd_index_splat<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::Index,
            ) -> Self::SimdIndex<S> {
                $simd_index_splat
            }

            #[inline]
            fn simd_index_add<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                $simd_index_add
            }

            #[inline]
            fn simd_index_less_than<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdIndex<S>,
                rhs: Self::SimdIndex<S>,
            ) -> Self::SimdMask<S> {
                $simd_index_less_than
            }

            #[inline]
            fn simd_and_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                $simd_and_mask
            }

            #[inline]
            fn simd_or_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                lhs: Self::SimdMask<S>,
                rhs: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                $simd_or_mask
            }

            #[inline]
            fn simd_not_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> Self::SimdMask<S> {
                $simd_not_mask
            }

            #[inline]
            fn simd_first_true_mask<S: faer_traits::pulp::Simd>(
                ctx: &Self::SimdCtx<S>,
                value: Self::SimdMask<S>,
            ) -> usize {
                $simd_first_true_mask
            }
        }

        impl faer_traits::RealField for $real {
            #[inline]
            fn epsilon_impl() -> Self {
                $epsilon
            }

            #[inline]
            fn nbits_impl() -> usize {
                $nbits
            }

            #[inline]
            fn min_positive_impl() -> Self {
                $min_positive
            }

            #[inline]
            fn max_positive_impl() -> Self {
                $max_positive
            }

            #[inline]
            fn sqrt_min_positive_impl() -> Self {
                $sqrt_min_positive
            }

            #[inline]
            fn sqrt_max_positive_impl() -> Self {
                $sqrt_max_positive
            }
        }
    };
}
