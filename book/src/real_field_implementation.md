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
1. Implement [`linalg_traits::real_field::RealFieldBase`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.Base.html).
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

### Checking that the trait is implemented

We can check that the [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) trait is even implemented in the first place using the [`linalg_traits::verify_trait_implemented!` macro](https://docs.rs/linalg-traits/latest/linalg_traits/macro.verify_trait_implemented.html).

### Checking disambiguation TODO

### TODO list other tests here

### TODO checking simd stuff

