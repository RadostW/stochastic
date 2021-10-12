import jax
import pytest
from pychastic.sde_solver import SDESolver
from pychastic.sde_problem import SDEProblem
import numpy as np

a = 1
b = 1
scalar_geometric_bm = SDEProblem(
  a = lambda x: a*x,
  b = lambda x: b*x,
  x0 = 1.0,
  tmax = 1.0,
  exact_solution = lambda x0, t, w: x0*np.exp((a-0.5*b*b)*t+b*w)   
)

a = 1.0
scalar_arctan_problem = SDEProblem(
  a = lambda x: -a**2*jax.numpy.sin(x)*jax.numpy.cos(x)**3,
  b = lambda x: a*jax.numpy.cos(x)**2,
  x0 = 0.0,
  tmax = 1.0,
  exact_solution = lambda x0, t, w: np.arctan(a*w+np.tan(x0))
)

polar_random_walk_problem = SDEProblem(
  
)

@pytest.mark.parametrize('solver,problem,steps,quantile_99', [
  (SDESolver(), scalar_geometric_bm, 2**7, 1),
  (SDESolver(), scalar_arctan_problem, 2**7, 0.09),
  (SDESolver(scheme='milstein'), scalar_geometric_bm, 2**7, 0.24),
  (SDESolver(scheme='milstein'), scalar_arctan_problem, 2**7, 0.008),
])
def test_again_exact_solution(solver: SDESolver, problem, steps, quantile_99):
  solver.dt = problem.tmax / steps

  result = solver.solve(problem)
  time_values = result['time_values'].reshape(-1, 1)
  solution_values = result['solution_values']
  wiener_values = result['wiener_values']

  exact_values = problem.exact_solution(problem.x0, time_values, wiener_values)
  errors = (exact_values - solution_values)
  end_error = errors[-1]
  assert abs(end_error) < quantile_99  # .99 quantile
