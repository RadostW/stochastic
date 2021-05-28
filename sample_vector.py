import pychastic
import numpy as np

problem = pychastic.sde_problem.VectorSDEProblem(
    lambda x: np.array([x[0]**3 - 3*x[0]*x[1]**2, -x[1]**3 + 3*x[1]*x[0]**2]),
    lambda x: np.array([[x[0]**2 - x[1]**2, 2*x[0]*x[1]]]).T,
    2,
    1,
    np.array([1.5, 0.5]),
    1,
)
solver = pychastic.sde_solver.VectorSDESolver()
solver.scheme = "milstein"
solver.solve(problem)
