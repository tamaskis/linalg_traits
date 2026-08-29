# Real Field

<!-- toc -->

## Background

The [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) trait describes real number types.

In mathematics, the real numbers form a **field**, meaning that they support the usual arithmetic operations of addition, subtraction, multiplication, and division (except division by zero). A real field also has an ordering, allowing values to be compared and concepts such as positive and negative numbers to be defined.

In practice, double-precision floating-point numbers (`f64` in Rust) are commonly used to represent real numbers. Other types are also used, although less frequently, such as single-precision floating-point numbers (`f32` in Rust). Linear algebra crates like [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/) and [`faer`](https://docs.rs/faer/latest/faer/) define their routines generically so they can work with any type satisfying the properties of a real field. They express these requirements via the [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html) and [`faer_traits::RealField`](https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html) traits, respectively.

Similarly, [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) provides an abstraction over concrete real-valued numeric types, allowing linear algebra code to work generically with them. However, it goes one step further by allowing linear algebra code (or any code using linear algebra types) to be written in an even more generic fashion, supporting multiple "backend" crates (e.g. [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/) and [`faer`](https://docs.rs/faer/latest/faer/)) while maintaining a uniform interface. The [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) is the first step toward this goal; any type satisfying its requirements will also satisfy the corresponding requirements for real field types in these backend crates.

This trait serves as the foundational interface for real number types that is cross-compatible across different linear algebra and numerical computation crates. Consequently, it originated as a trait that defined the superset of functionality provided by [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html) and [`faer_traits::RealField`](https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html). However, real numbers are used well beyond these specific crates, with many common operations not required by these crates (for example, [`faer_traits::RealField`](https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html) doesn't require that a real number implement a `sin` method, which is one of the most widely used mathematical functions). As a result, this trait defines the superset (mostly; there are some methods missing) of functionality provided by [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html), [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html), and [`faer_traits::RealField`](https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html).

## Additional features geared towards automatic differentiation

While one aim of the `linalg-traits` crate is to define a generic interface for linear algebra types, another is to provide the scaffolding needed for implemeneting forward-mode automatic differentiation. This secondary goal adds a lot of additional constraints/requirements to the definition of the [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) trait.

### Brief overview of forward-mode automatic differentiation

Forward-mode automatic differentiation is a technique for "automatically" obtaining the derivative of an arbitrary function. The simplest case is differentiating a univariate, scalar-valued function $f:\mathbb{R}\to\mathbb{R}$. In Rust, we would typically write such as function as

```rust
fn f(x: f64) -> f64 {
    ...
}
```

To support forward-mode automatic differentiation, we would instead write this function as

```rust
fn f<R>(x: R) -> R where R: RealField {
    ...
}
```

This is so that:

1. We can still pass in an `f64` and get an `f64` out.
1. We can pass in a [`Dual`](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html) and get a [`Dual`](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html) out, where the `dual`[^1] part of the return value stores the derivative of `f`.

[^1]: Where `Dual { real: f64, dual: f64 }`.

> [!NOTE]
> The [`numdiff`](https://docs.rs/numdiff/latest/numdiff) crate implements the [`numdiff::Dual`](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html) (for 1st-order differentiation) and [`numdiff::HyperDual`](https://docs.rs/numdiff/latest/numdiff/struct.HyperDual.html) (for 2nd-order differentiation) types and also has many convenient macros for performing forward-mode automatic differentiation, including for vector-valued and multivariate functions.

### What this means for `linalg_traits::RealField`

Making functions compatible with forward-mode automatic differentiation is already burdensome for users; they have to replace all concrete types in function signatures with generic types implementing traits provided by this crate. To minimize the pain, we wanted [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) to:

1. Define the superset of all operations supported by `f64`, `num_traits::Float`, `nalgebra::RealField`, and `faer_traits::RealField`.
    * _This allows us to write code in a way where it will always work, regardless of what third-party backend crates are used._
1. Have maximum interoperability with `f64` (e.g. support arithmetic operations between `f64` and `T: RealField`, cast `T: RealField` to/from `f64`, etc).
    * _This allows us to write generic code in a way where we don't have to do a lot of manual casting to/from `f64`._
1. Not introduce additional [trait disambiguation](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html) burden on the user.
    * _This has similar benefit to two items above, where the generic code we write looks nearly identical to the code we write just using the concrete `f64` type._
    * _See the ["More on trait disambiguation" section below](#more-on-trait-disambiguation)_.
  
#### More on trait disambiguation

> [!NOTE]
> The information provided below is a description of how we circumvent trait disambiguation in the design of [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html). However, users of this crate should not have to ever worry about this, unless they are manually implementing the [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) trait for a custom type in the (unrecommended) method described [here](https://tamaskis.github.io/linalg_traits/real_field_implementation.html#alternative-manual-implementation).

As an example, when the `nalgebra` feature is enabled, a large number of methods documented under the [`Methods` page of the `linalg-traits` book](https://tamaskis.github.io/linalg_traits/real_field_methods.html) are provided directly by [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html) and/or its super traits. When the `nalgebra` feature is NOT enabled, these same methods would be provided by [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html) by essentially forwarding the corresponding [`linalg_traits::real_field::RealFieldBase`](https://docs.rs/linalg-traits/latest/linalg_traits/real_field/trait.RealFieldBase.html) method.

We _could_ provide both implementations simultaneously; for example we could simultaneously have `nalgebra::ComplexField::sin` (where [`nalgebra::ComplexField`](https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html) is a super trait of [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html)) and `linalg_traits::RealField::sin` implemented for some concrete type, but then we would have to disambiguate the trait every time this method was called. For example, we would have to do

```rust,ignore
<T as linalg_traits::RealField>::sin(1.0)  // or <T as nalgebra::ComplexField>::sin(1.0)
```

instead of

```rust,ignore
1.0.sin() // or equivalently `T::sin(1.0)`, where T: RealField
```

Therefore, when the `nalgebra` feature is enabled, the methods of this trait that collide with [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html) (and any of its supertraits') method names are omitted here to avoid ambiguous method resolution.

Identical approaches are taken for any other name clashes introduced by other features.