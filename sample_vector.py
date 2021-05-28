import pychastic
import numpy as np
problem = pychastic.sde_problem.VectorSDEProblem(lambda x: np.array([1.,1.]), lambda x: np.array([[1.0,0.5],[0.5,1.0]]), 2, 2, np.array([1.5,0.5]), 1)
solver = pychastic.sde_solver.VectorSDESolver()
solver.scheme = 'euler'
solver.solve(problem)
