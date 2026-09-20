# Implementation

<!-- toc -->

## Background

[`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) is a collection of a complicated and nuanced set of supertraits where different implementations of the same methods are conditionally compiled based on what features are enabled. 

There are two ways one could implement it:

1. Figure out the full set of traits needed from the complex [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) definition and manually implement them one by one (**_not_ recommended**).
1. Manually implement [`linalg_traits::real_field::Base`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.Base.html) and [`linalg_traits::real_field::RealFieldBase`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.RealFieldBase.html), and let [`linalg_traits::impl_real_field!`](https://docs.rs/linalg-traits/latest/linalg_traits/macro.impl_real_field.html) take care of the rest (**recommended approach**).

## Implementation approaches

### Recommended implementation

1. Implement [`linalg_traits::real_field::Base`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.Base.html).
1. Implement [`linalg_traits::real_field::RealFieldBase`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.RealFieldBase.html).
1. Call [`linalg_traits::impl_real_field!`](https://docs.rs/linalg-traits/latest/linalg_traits/macro.impl_real_field.html) on your type.
  - _Based on what you've defined for [`linalg_traits::real_field::Base`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.Base.html) and [`linalg_traits::impl_real_field!`](https://docs.rs/linalg-traits/latest/linalg_traits/macro.impl_real_field.html), this will take care of implementing everything else needed for [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html)._

> [!TIP]
> [`linalg-traits-example`](https://docs.rs/linalg-traits/latest/linalg_traits_example/) has an example implementation of [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) for [`MyReal {x: f64}`](https://docs.rs/linalg-traits/latest/linalg_traits_example/my_real/struct.MyReal.html) using this approach (you can also browse the source code [here](https://github.com/tamaskis/linalg_traits/tree/main/linalg_traits_example/example/src/)).

### Alternative (manual) implementation

One _could_ check out the [`linalg_traits::RealField` docs](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) and trace through all the required traits and implement them one-by-one, but this would be a difficult endeavour.

> [!CAUTION]
> The [`linalg_traits::RealField` docs](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) are generated with **all** features enabled, so you may not even be able to correctly trace through the traits that are required for your specific setup.
> Furthermore, a very rich set of arithmetic operators is required, and how this is defined via trait bounds is also quite complex and difficult to manually read and reason about. Even if you were able to figure out every single operator that has to be implemented, it would be arduous to manually implement every single variation.

## Testing the implementation

> [!TIP]
> Use the example crate as the main reference for validating your own implementation:
>
> - [`linalg_traits_example`](https://docs.rs/linalg-traits/latest/linalg_traits_example/) provides a custom scalar, [`MyReal`](https://docs.rs/linalg-traits/latest/linalg_traits_example/my_real/struct.MyReal.html), that forwards each method to its wrapped `f64` value.
> - The runtime validation lives in [`linalg_traits_example/src/real_field_base.rs`](https://github.com/tamaskis/linalg_traits/blob/main/linalg_traits_example/src/real_field_base.rs) and is the clearest model for behavior checks on `RealFieldBase`.
> - The generic no-disambiguation check is in [`tests/real_field_methods_no_disambiguation.rs`](https://github.com/tamaskis/linalg_traits/blob/main/tests/real_field_methods_no_disambiguation.rs), and the `nalgebra`-specific qualified-call example is in [`examples/real_field_abs.rs`](https://github.com/tamaskis/linalg_traits/blob/main/examples/real_field_abs.rs).
> - Backend sanity checks live in [`tests/faer_complex_field.rs`](https://github.com/tamaskis/linalg_traits/blob/main/tests/faer_complex_field.rs) and [`tests/nalgebra_complex_field.rs`](https://github.com/tamaskis/linalg_traits/blob/main/tests/nalgebra_complex_field.rs).

### Checking that the trait is implemented

We can check that the [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) trait is even implemented in the first place using the [`linalg_traits::verify_trait_implemented!` macro](https://docs.rs/linalg-traits/latest/linalg_traits/macro.verify_trait_implemented.html).

For example, the example crate checks both layers of the recommended implementation:

```rust,ignore
use linalg_traits::{RealField, verify_trait_implemented};
use linalg_traits::real_field::RealFieldBase;

const _: bool = verify_trait_implemented!(MyReal: RealFieldBase);

// After calling `impl_real_field!`:
const _: bool = verify_trait_implemented!(MyReal: RealField);
```

These checks are compile-time checks. They do not execute at runtime, so a successful
`cargo check` or `cargo test` is enough to verify that the required trait bounds are
satisfied.

### Checking disambiguation

When a backend feature is enabled, some methods may also be provided by a backend
trait. The `RealField` definition omits its own colliding methods where necessary, so
normal generic calls should remain unambiguous:

```rust,ignore
fn abs<T: linalg_traits::RealField>(value: T) -> T {
  value.abs()
}
```

The `nalgebra` feature is an intentional exception for `abs`: `num_traits::Signed`
and `simba::scalar::ComplexField` both provide a method with that name. If code needs
to select a particular implementation, use fully qualified syntax:

```rust,ignore
let a = value.abs();
let b = <T as num_traits::Signed>::abs(&value);
assert_eq!(a, b);
```

The [`assert_no_trait_disambiguation_needed`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_no_trait_disambiguation_needed.html)
helper calls the complete `RealField` method surface on a generic `T: RealField` to ensure no
backend trait method needs explicit qualification. The crate also includes the integration test
that exercises the same check:

```rust,ignore
use linalg_traits::RealField;
use linalg_traits::real_field::assert_no_trait_disambiguation_needed;

assert_no_trait_disambiguation_needed::<f64>(1.0, 2.0, 3.0);
```

The [`real_field_abs.rs`](https://github.com/tamaskis/linalg_traits/blob/main/examples/real_field_abs.rs)
example shows the corresponding qualified call for the `nalgebra` case.

### Testing operator overloads and conversions

Once the trait implementation compiles, the next step is to validate the runtime behavior delegated through the trait. The crate provides reusable helpers for this in `linalg_traits::real_field`:

- [`assert_base`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_base.html) checks the semantics of the base scalar API: `Default`, `Debug`, `Display`, `Copy`/`Clone`, and the expected default value.
- [`assert_real_field_operations`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_real_field_operations.html) checks the core arithmetic, comparison, and assignment surface for `T` and `&T` operands.
- [`assert_f64_conversions`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_f64_conversions.html) checks conversions to and from `f64`.
- [`assert_f64_lhs_ops`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_f64_lhs_ops.html) checks operations where the `f64` operand appears on the left-hand side.
- [`assert_f64_rhs_ops`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/fn.assert_f64_rhs_ops.html) checks operations where the `f64` operand appears on the right-hand side.

These helpers are useful for both `f64` and custom scalar types because they exercise the same generic dispatch patterns that downstream code will use. For example:

```rust,ignore
use linalg_traits::real_field::{
    assert_base,
    assert_f64_conversions,
    assert_f64_lhs_ops,
    assert_f64_rhs_ops,
    assert_real_field_operations,
};

assert_base(MyReal::new(1.5), MyReal::new(0.0), "1.5", "1.5");

assert_real_field_operations(
    MyReal::new(7.0),
    MyReal::new(2.0),
    MyReal::new(-7.0),
    MyReal::new(9.0),
    MyReal::new(5.0),
    MyReal::new(14.0),
    MyReal::new(3.5),
    MyReal::new(1.0),
    false,
    false,
    false,
    true,
    true,
);

assert_f64_conversions(MyReal::new(2.0), 2.0, 2.0, MyReal::new(2.0));

assert_f64_rhs_ops(
    MyReal::new(3.0),
    2.0,
    false,
    false,
    false,
    true,
    true,
    MyReal::new(5.0),
    MyReal::new(1.0),
    MyReal::new(6.0),
    MyReal::new(1.5),
    MyReal::new(1.0),
);

assert_f64_lhs_ops(
    3.0,
    MyReal::new(2.0),
    false,
    false,
    false,
    true,
    true,
    MyReal::new(5.0),
    MyReal::new(1.0),
    MyReal::new(6.0),
    MyReal::new(1.5),
    MyReal::new(1.0),
);
```

The exact numeric values should match your implementation's semantics, but the point is to exercise every overload family rather than checking just a single operation in isolation.

### Testing `RealFieldBase` implementation

This is the part you should test manually. After you have written the generic behavior checks above for your custom type, you should also exercise every custom method implemented in your `RealFieldBase` implementation. That is the code you own, and its semantics are specific to your type.

### Other useful tests

For backend (e.g. `nalgebra`, `faer`) integrations, you do not need to hand-write exhaustive tests for every method generated by `impl_real_field!`. Those backend trait implementations are generated automatically from your `RealFieldBase` implementation and should be treated as library-supplied machinery.

The repository already includes a few backend sanity checks for the generated adapters, confirming that the forwarded methods agree with the base implementation:

- [`tests/faer_complex_field.rs`](https://github.com/tamaskis/linalg_traits/blob/main/tests/faer_complex_field.rs)
- [`tests/nalgebra_complex_field.rs`](https://github.com/tamaskis/linalg_traits/blob/main/tests/nalgebra_complex_field.rs)

These are useful validation checks, but they are not a substitute for testing the custom `RealFieldBase` behavior you wrote yourself.


### Checking SIMD behavior

`impl_real_field!` gives custom scalar types a scalar-only backend implementation. It
does not reinterpret a custom type as `f64`, because doing so could discard state such
as derivatives carried by a dual number. With the `faer` feature, the generated
`faer_traits::ComplexField` implementation therefore sets
`SIMD_CAPABILITIES` to `faer_traits::SimdCapabilities::Copy`, which makes `faer` use
scalar kernels.

This is not a claim that the type has vectorized arithmetic. SIMD entry points are
provided only to satisfy the backend trait and should not be reached when scalar
kernels are selected. Verify the contract directly:

```rust,ignore
assert_eq!(
  <MyReal as faer_traits::ComplexField>::SIMD_CAPABILITIES,
  faer_traits::SimdCapabilities::Copy,
);
```

The `nalgebra` implementation similarly provides a one-lane
`simba::simd::SimdValue`; it is a scalar representation, not a promise of hardware
SIMD.