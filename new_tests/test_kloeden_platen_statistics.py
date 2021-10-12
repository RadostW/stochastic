import pychastic
import numpy as np

def test_kp_exercise_9_3_3():
  a = 1.5
  b = 1.0
  problem = pychastic.sde_problem.SDEProblem(
    lambda x: a*x,
    lambda x: b*x,
    x0 = 1.0,
    tmax=1,
    exact_solution = lambda x0, t, w: x0*np.exp((a-b**2/2)*t+b*w)
  )

  solver = pychastic.sde_solver.SDESolver()

  dts = [2**-4, 2**-5, 2**-6, 2**-7]
  n_rows = 10
  n_wieners_per_cell = 25

  results = np.zeros((len(dts), n_rows*n_wieners_per_cell), dtype=dict)  # dt x wiener

  for n_dt, dt in enumerate(dts):
    solver.dt = dt
    solutions = solver.solve_many(problem, n_trajectories=n_rows*n_wieners_per_cell)
    results[n_dt] = solutions
  
  results = results.reshape(-1, n_rows, n_wieners_per_cell)

  def get_end_error(result):
    x = result['solution_values'][-1]
    x_exact = problem.exact_solution(problem.x0, result['time_values'][-1], result['wiener_values'][-1])
    return abs(x - x_exact)

  errors = np.array([get_end_error(res) for res in results.flatten()]).reshape(results.shape)
  table = errors.mean(axis=-1).T
  print(table.shape)