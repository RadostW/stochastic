import jax
from pychastic.sde_problem import SDEProblem

def test_scalar_input():
  tmax = 1
  x0 = 0
  a = lambda x: jax.numpy.array(0)
  b = lambda x: jax.numpy.array(0)
  SDEProblem(a, b, x0, tmax)

def test_vector_input():
  tmax = 1
  dimension = 2
  noise_terms = 3
  x0 = jax.numpy.zeros(dimension)
  a = lambda x: jax.numpy.zeros(dimension)
  b = lambda x: jax.numpy.zeros((dimension, noise_terms))
  SDEProblem(a, b, x0, tmax)
