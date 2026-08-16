# Interface Methods

<!-- toc -->

## `as_slice` (view as a slice of `f64`)

> [!NOTE]
> This method is primarily used for deriving [`approx::UlpsEq`](https://docs.rs/approx/latest/approx/trait.UlpsEq.html) for types with multiple components that are implementing [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) (for example, this would be used for something like a [dual number](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html) that is substituted in place of a single real number for forward mode automatic differentiation).

See [`linalg_traits::RealField::as_slice`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html#method.as_slice).

```rust
{{#include ../../examples/real_field_as_slice.rs}}
```
