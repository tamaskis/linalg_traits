use crate::dual::Dual;
use crate::scalar::Scalar;

#[macro_export]
macro_rules! sderivative_generic {
    ($f:expr, $x0:expr) => {{
        // let x = Dual::new($x0, 1.0);
        let x = Dual::new($x0, 1.0);
        let f_eval = $f(x);
        f_eval.get_dual()
    }};
}

macro_rules! create_caller {
    ($func_name:ident, $generic_func:ident) => {
        fn $func_name<S: Scalar>(value: S) -> f64 {
            // Cast the value to a different concrete type (T -> ConcreteTypeB)
            let temp_value = Dual::new(value.to_f64().unwrap(), 1.0);

            // Call the passed-in generic function with a reference to the new concrete type
            let result = $generic_func(temp_value); // Now using ConcreteTypeB

            // Since temp_value is ConcreteTypeB, we need to cast it back to ConcreteTypeA first
            result.get_dual() // Convert back to ConcreteTypeA
        }
    };
}

// fn helper1(func: &impl Fn(Dual) -> Dual) -> Box<dyn Fn(Dual) -> Dual + '_> {
//     Box::new(move |x: Dual| func(x))
// }

// fn helper<S: Scalar>(func: &impl Fn(S) -> S) -> Box<dyn Fn(Dual) -> Dual + '_> {
//     Box::new(move |x: Dual| func(x))
// }

// pub struct Differentiator {}

// impl Differentiator {
//     pub fn sderivative<S: Scalar>(f: &impl Fn(S) -> S, x0: S) -> Box<dyn Fn(Dual) -> Dual> {
//     Box::new(move |d: Dual| Dual {
//         re: f(d.re),          // Apply the function to the real part
//         im: d.im * f(d.re),   // Adjust the dual part based on the derivative
//     })
// }

#[cfg(test)]
mod tests {
    use crate::dual::Dual;
    use crate::scalar::Scalar;
    // use crate::vector::vector_trait::Vector;

    // macro_rules! vderivative {
    //     ($f:expr, $x0:expr) => {{
    //         let x = Dual::new($x0, 1.0);
    //         let f_eval = $f(x);
    //         // might work but need to swap internal type
    //         for i in 0..f_eval.len() {}
    //         f_eval.get_dual()
    //     }};
    // }

    pub fn func<S: Scalar>(x: S) -> S {
        x.powi(2)
    }

    // pub fn func2<S: Scalar>(x: S) -> Vec<S> {
    //     vec![x.sin(), x.cos()]
    // }

    // #[test]
    // fn example() {
    //     let result = sderivative!(func, 5.0);
    //     assert_eq!(result, 10.0);
    // }

    // #[test]
    // fn example_2() {
    //     let result = vderivative!(func2, 5.0);
    //     assert_eq!(result, 10.0);
    // }

    #[test]
    fn example_3() {
        create_caller!(derivative, func);
        println!("{:?}", derivative(5.0));
        assert_eq!(derivative(5.0), 10.0);
    }
}
