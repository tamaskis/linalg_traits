# Scalar

At its core, the `Scalar` trait is an extension of the [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html) trait and is meant to represent real numbers. The additional restrictions it imposes are:

* implemention of arithmetic-assignment operations (i.e. `+=`, `-=`, `*=`, `/=`, and `%=`)
* interoperability with `f64` (i.e. `S + f64`, `S - f64`, `S * f64`, `S / f64`, `S % f64`, `S += f64`, `S -= f64`, `S *= f64`, `S /= f64`, `S %= f64`, where `S` is any type implementing the `Scalar` trait)
* casting to/from `f64` (i.e. `From<f64>`, `Into<f64>`)
* `'static` TODO only needed for ndarray

The primary motivation for the `Scalar` trait is for use by the `numdiff` crate for supporting forward-mode automatic differentation. This type of autodiff relies on a generic programming approach, where the functions we are differentiating are written in terms of some generic scalar type `S`. Most of the time, when you're evaluating such functions, you just input an `f64` and get back and `f64`. However, autodiff backends will input dual numbers into these same functions and get out dual numbers which contain derivative information.

To support forward-mode automatic differentation, the `numdiff` crate implements the `Scalar` trait for the `Dual` and `HyperDual` types. These types aren't real numbers, so it would be confusing if this trait were instead called `Real` and then we implemented it for a non-real number.

There are multiple popular crates that support matrix/linear algebra (e.g. `nalgebra`, `faer`, `ndarray`). Each of these crates have different matrix/array types, and the traits that the elements of those matrix/array types have to satisfy also changes crate-to-crate.

The trait bounds we impose on the `Scalar` trait are designed to satisfy the following objectives:
* minimize code differences between writing code for `f64` vs. a generic `S: Scalar`
* be compatible with major linear algebra crates (specifically `nalgebra`, `faer`, and `ndarray`)
* support forward-mode automatic differentiation

## Implementing the `Scalar` trait

Implementing the `Scalar` trait is a nontrivial task that requires implementing an extensive set of traits. The full set of traits that have to be manually implemented is:

### [`std::ops::Add<Rhs, Output = Output>`](https://doc.rust-lang.org/std/ops/trait.Add.html)

Required by:

* [`num_traits::NumOps`](https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html)
  * [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
    * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
      * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
        * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::ops::Sub<Rhs, Output = Output>`](https://doc.rust-lang.org/std/ops/trait.Sub.html)

Required by:

* [`num_traits::NumOps`](https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html)
  * [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
    * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
      * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
        * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::ops::Mul<Rhs, Output = Output>`](https://doc.rust-lang.org/std/ops/trait.Mul.html)

Required by:

* [`num_traits::NumOps`](https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html)
  * [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
    * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
      * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
        * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::ops::Div<Rhs, Output = Output>`](https://doc.rust-lang.org/std/ops/trait.Div.html)

Required by:

* [`num_traits::NumOps`](https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html)
  * [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
    * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
      * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
        * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::ops::Rem<Rhs, Output = Output>`](https://doc.rust-lang.org/std/ops/trait.Rem.html)

Required by:

* [`num_traits::NumOps`](https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html)
  * [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
    * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
      * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
        * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::cmp::PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html)

Required by:

* [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
  * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
    * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
      * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`nalgebra::base::Scalar`](https://docs.rs/nalgebra/latest/nalgebra/base/trait.Scalar.html) _(when `nalgebra` feature is selected)_
  * [`linalg_traits::NalgebraScalar](TODO) _(when `nalgebra` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `nalgebra` feature is selected)_

### [`num_traits::Zero`](https://docs.rs/num-traits/latest/num_traits/identities/trait.Zero.html)

Required by:

* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_
* [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
  * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
    * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
      * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`num_traits::One`](https://docs.rs/num-traits/latest/num_traits/identities/trait.One.html)

Required by:

* [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
  * [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
    * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
      * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`num_traits::Num`](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)

Required by:

* [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
  * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
    * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`num_traits::cast::NumCast`](https://docs.rs/num-traits/latest/num_traits/cast/trait.NumCast.html)

Required by:

* [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
  * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
    * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::cmp::PartialOrd`](https://doc.rust-lang.org/std/cmp/trait.PartialOrd.html)

Required by:

* [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
  * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
    * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Neg<Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Neg.html)

Required by:

* [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
  * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
    * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::marker::Copy`](https://doc.rust-lang.org/std/marker/trait.Copy.html)

Required by:

* [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html)
  * [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
    * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::clone::Clone`](https://doc.rust-lang.org/std/clone/trait.Clone.html)

Required by:

* [`nalgebra::base::Scalar`](https://docs.rs/nalgebra/latest/nalgebra/base/trait.Scalar.html) _(when `nalgebra` feature is selected)_
  * [`linalg_traits::NalgebraScalar](TODO) _(when `nalgebra` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `nalgebra` feature is selected)_


### [`std::fmt::Debug`](https://doc.rust-lang.org/std/fmt/trait.Debug.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`nalgebra::base::Scalar`](https://docs.rs/nalgebra/latest/nalgebra/base/trait.Scalar.html) _(when `nalgebra` feature is selected)_
  * [`linalg_traits::NalgebraScalar](TODO) _(when `nalgebra` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `nalgebra` feature is selected)_

### [`'static`](https://doc.rust-lang.org/rust-by-example/scope/lifetime/static_lifetime.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
* [`nalgebra::base::Scalar`](https://docs.rs/nalgebra/latest/nalgebra/base/trait.Scalar.html) _(when `nalgebra` feature is selected)_
  * [`linalg_traits::NalgebraScalar](TODO) _(when `nalgebra` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `nalgebra` feature is selected)_
* [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) _(when `ndarray` feature is selected)_
  * [`linalg_traits::NdarrayScalar] _(when `ndarray` feature is selected)_
    * [`linalg_traits::Scalar](TODO) _(when `ndarray` feature is selected)_

### [`std::ops::AddAssign<Self>`](https://doc.rust-lang.org/std/ops/trait.AddAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::SubAssign<Self>`](https://doc.rust-lang.org/std/ops/trait.SubAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::MulAssign<Self>`](https://doc.rust-lang.org/std/ops/trait.MulAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::DivAssign<Self>`](https://doc.rust-lang.org/std/ops/trait.DivAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::RemAssign<Self>`](https://doc.rust-lang.org/std/ops/trait.RemAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Add<f64, Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Add.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Sub<f64, Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Sub.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Mul<f64, Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Mul.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Div<f64, Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Div.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::Rem<f64, Output = Self>`](https://doc.rust-lang.org/std/ops/trait.Rem.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::AddAssign<f64>`](https://doc.rust-lang.org/std/ops/trait.AddAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::SubAssign<f64>`](https://doc.rust-lang.org/std/ops/trait.SubAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::MulAssign<f64>`](https://doc.rust-lang.org/std/ops/trait.MulAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::DivAssign<f64>`](https://doc.rust-lang.org/std/ops/trait.DivAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::ops::RemAssign<f64>`](https://doc.rust-lang.org/std/ops/trait.RemAssign.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::convert::From<f64>`](https://doc.rust-lang.org/std/convert/trait.From.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

### [`std::convert::Into<f64>`](https://doc.rust-lang.org/std/convert/trait.Into.html)

Required by:

* [`linalg_traits::RealScalarBase`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealScalarBase.html)
  * [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)

## Additional traits required for `faer`

| Trait | Reason | Trait(s) requiring this trait |
| ----- | ------ | ----------------------------- |
| `faer_traits::ComplexField` | TODO | `faer_traits::RealField` |
| `faer_traits::RealField` | TODO | `linalg_traits::Scalar` |

## Additional traits required for `ndarray`

| Trait | Reason |
| ----- | ------ |
| `'static` | This is required by the [`ndarray::ScalarOperand`](https://docs.rs/ndarray/latest/ndarray/trait.ScalarOperand.html) marker trait and the [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) trait. |
| [`ndarray::ScalarOperand`](https://docs.rs/ndarray/latest/ndarray/trait.ScalarOperand.html) | This has to be implemented by elements of an array so that mixed scalar/matrix operations can be performed (e.g. `a + 1.0f64`, where `a` is a matrix, i.e. `ndarray::Array2<f64>`) |
| [`ndarray::LinalgScalar`](https://docs.rs/ndarray/latest/ndarray/trait.LinalgScalar.html) | This makes elements of an `ndarray` array compatible with linear algebra operations. **Note:** In general, this does not need to be manually implemented; a blanket implementation is provided. |