# Interoperability with f64

<!-- toc -->

> [!NOTE]
> Assignment operations with `f64` on the left-hand side (e.g. `f64 += T`) are intentionally **not** supported. See the [`RealFieldOperations`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealFieldOperations.html) documentation for the rationale.

## Conversion

The following conversions are supported for `T: RealField`:

* `f64::from(T)` / `T::into(self) -> f64`
* `T::from(f64)` / `f64::into(self) -> T`

**Example**

```rust
{{#include ../../examples/real_field_f64_conversion.rs}}
```

## Equality

The following equality operations are supported for `T: RealField`:

* `T == f64`
* `T != f64`
* `f64 == T`
* `f64 != T`

## Ordering

The following ordering operations are supported for `T: RealField`:

* `T < f64`
* `T <= f64`
* `T > f64`
* `T >= f64`
* `f64 < T`
* `f64 <= T`
* `f64 > T`
* `f64 >= T`

## Addition

The following addition operations are supported for `T: RealField`:

* `T + f64`
* `T + &f64`
* `&T + f64`
* `&T + &f64`
* `f64 + T`
* `f64 + &T`
* `&f64 + T`
* `&f64 + &T`

**Example**

```rust
{{#include ../../examples/real_field_f64_addition.rs}}
```

## Subtraction

The following subtraction operations are supported for `T: RealField`:

* `T - f64`
* `T - &f64`
* `&T - f64`
* `&T - &f64`
* `f64 - T`
* `f64 - &T`
* `&f64 - T`
* `&f64 - &T`

**Example**

```rust
{{#include ../../examples/real_field_f64_subtraction.rs}}
```

## Multiplication

The following multiplication operations are supported for `T: RealField`:

* `T * f64`
* `T * &f64`
* `&T * f64`
* `&T * &f64`
* `f64 * T`
* `f64 * &T`
* `&f64 * T`
* `&f64 * &T`

**Example**

```rust
{{#include ../../examples/real_field_f64_multiplication.rs}}
```

## Division

The following division operations are supported for `T: RealField`:

* `T / f64`
* `T / &f64`
* `&T / f64`
* `&T / &f64`
* `f64 / T`
* `f64 / &T`
* `&f64 / T`
* `&f64 / &T`

**Example**

```rust
{{#include ../../examples/real_field_f64_division.rs}}
```

## Remainder after division

The following remainder after division operations are supported for `T: RealField`:

* `T % f64`
* `T % &f64`
* `&T % f64`
* `&T % &f64`
* `f64 % T`
* `f64 % &T`
* `&f64 % T`
* `&f64 % &T`

**Example**

```rust
{{#include ../../examples/real_field_f64_remainder.rs}}
```

## Addition-assignment

The following addition-assignment operations are supported for `T: RealField`:

* `T += f64`
* `T += &f64`

**Example**

```rust
{{#include ../../examples/real_field_f64_add_assign.rs}}
```

## Subtraction-assignment

The following subtraction-assignment operations are supported for `T: RealField`:

* `T -= f64`
* `T -= &f64`

**Example**

```rust
{{#include ../../examples/real_field_f64_sub_assign.rs}}
```

## Multiplication-assignment

The following multiplication-assignment operations are supported for `T: RealField`:

* `T *= f64`
* `T *= &f64`

**Example**

```rust
{{#include ../../examples/real_field_f64_mul_assign.rs}}
```

## Division-assignment

The following division-assignment operations are supported for `T: RealField`:

* `T /= f64`
* `T /= &f64`

**Example**

```rust
{{#include ../../examples/real_field_f64_div_assign.rs}}
```

## Remainder-assignment

The following remainder-assignment operations are supported for `T: RealField`:

* `T %= f64`
* `T %= &f64`

**Example**

```rust
{{#include ../../examples/real_field_f64_rem_assign.rs}}
```
