import jax
import pytest
from pychastic.sde_solver import SDESolver
from pychastic.sde_problem import SDEProblem
import numpy as np

a = 0.5
b = 1
scalar_geometric_bm = SDEProblem(
  a = lambda x: a*x,
  b = lambda x: b*x,
  x0 = 1.0,
  tmax = 1.0,
  exact_solution = lambda x0, t, w: x0*np.exp((a-0.5*b*b)*t+b*w)   
)

@pytest.mark.parametrize('solver', [SDESolver()])
@pytest.mark.parametrize('problem', [scalar_geometric_bm])
def test_again_exact_solution(solver: SDESolver, problem):
  steps = 10000
  solver.dt = problem.tmax / steps

  result = solver.solve(problem)
  time_values = result['time_values'].reshape(-1, 1)
  solution_values = result['solution_values']
  wiener_values = result['wiener_values']

  exact_values = problem.exact_solution(problem.x0, time_values, wiener_values)
  errors = (exact_values - solution_values)
  end_error = errors[-1]
  assert jax.numpy.isclose(end_error, 0, atol=1e-4, rtol=1e-2)
