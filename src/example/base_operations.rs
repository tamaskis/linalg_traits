use crate::example::my_scalar::MyScalar;
use crate::impl_real_field_operations;
use crate::real_field_operations::BaseOperations;

impl BaseOperations for MyScalar {
    fn neg(self) -> Self {
        MyScalar::new(-self.x)
    }
    fn add(self, rhs: Self) -> Self {
        MyScalar::new(self.x + rhs.x)
    }
    fn sub(self, rhs: Self) -> Self {
        MyScalar::new(self.x - rhs.x)
    }
    fn mul(self, rhs: Self) -> Self {
        MyScalar::new(self.x * rhs.x)
    }
    fn div(self, rhs: Self) -> Self {
        MyScalar::new(self.x / rhs.x)
    }
    fn rem(self, rhs: Self) -> Self {
        MyScalar::new(self.x % rhs.x)
    }
    fn eq(self, rhs: Self) -> bool {
        self.x == rhs.x
    }
    fn partial_cmp(self, rhs: Self) -> Option<std::cmp::Ordering> {
        self.x.partial_cmp(&rhs.x)
    }
}

impl_real_field_operations!(MyScalar);
