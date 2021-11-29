import pytest 
import jax.numpy as jnp

from Operator.operators import *

# matrix examples for int

matrix_int_a = jnp.random.randint(10, size=10)
matrix_int_a2 = jnp.random.randint(7, size=10)

matrix_int_b = jnp.random.randint(10, size=(2, 2))
matrix_int_b2 = jnp.random.randint(7, size=(2, 2))

matrix_int_c = jnp.random.randint(10, size=(3, 3))
matrix_int_c2 = jnp.random.randint(7, size=(3, 3))

matrix_int_d = jnp.random.randint(10, size=(4, 4))
matrix_int_d2 = jnp.random.randint(7, size=(4, 4))

matrix_int_e = jnp.random.randint(10, size=(2, 10))
matrix_int_e2 = jnp.random.randint(7, size=(2, 10))

matrix_int_f = jnp.random.randint(10, size=(10, 2))
matrix_int_f2 = jnp.random.randint(7, size=(10, 2))

matrix_int_g = jnp.random.randint(10, size=(2, 10, 10))
matrix_int_g2 = jnp.random.randint(7, size=(2, 10, 10))

matrix_int_h = jnp.random.randint(10, size=(10, 2, 10))
matrix_int_h2 = jnp.random.randint(7, size=(10, 2, 10))

matrix_int_i = jnp.random.randint(10, size=(10, 10, 2))
matrix_int_i2 = jnp.random.randint(7, size=(10, 10, 2))

# matrix examples for floats between 0 and 1

matrix_float_a = jnp.random.rand(10, 1)
matrix_float_a2 = jnp.random.rand(10, 1)

matrix_float_b = jnp.random.rand(2, 2)
matrix_float_b2 = jnp.random.rand(2, 2)

matrix_float_c = jnp.random.rand(3, 3)
matrix_float_c2 = jnp.random.rand(3, 3)

matrix_float_d = jnp.random.rand(4, 4)
matrix_float_d2 = jnp.random.rand(4, 4)

matrix_float_e = jnp.random.rand(2, 10)
matrix_float_e2 = jnp.random.rand(2, 10)

matrix_float_f = jnp.random.rand(10, 2)
matrix_float_f2 = jnp.random.rand(10, 2)

matrix_float_g = jnp.random.rand(12, 10, 10)
matrix_float_g2 = jnp.random.rand(2, 10, 10)

matrix_float_h = jnp.random.rand(10, 2, 10)
matrix_float_h2 = jnp.random.rand(10, 2, 10)

matrix_float_i = jnp.random.rand(10, 10, 2)
matrix_float_i2 = jnp.random.rand(10, 10, 2)

def covariance_case():
    """
    Covariance case
    """
    assert Operators.covariance(matrix_int_a, matrix_int_a2) == jnp.cov(matrix_int_a, matrix_int_a2)
    assert Operators.covariance(matrix_int_b, matrix_int_b2) == jnp.cov(matrix_int_b, matrix_int_b2)
    assert Operators.covariance(matrix_int_c, matrix_int_c2) == jnp.cov(matrix_int_c, matrix_int_c2)
    assert Operators.covariance(matrix_int_g, matrix_int_g2) == jnp.cov(matrix_int_g, matrix_int_g2)