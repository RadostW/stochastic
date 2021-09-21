import unittest
from numpy.lib.shape_base import split
from sortedcontainers.sorteddict import SortedDict

import pychastic

import random       # make tests deterministic
import numpy as np  #
random.seed(0)      #
np.random.seed(0)   #

class TestEulerMaruyama(unittest.TestCase):
    def test_euler_error_size(self):
        
        a = 1.5
        b = 0.1
        x0 = 1.0
        tmax = 1.0

        problem = pychastic.sde_problem.SDEProblem(
            lambda x: a*x,
            lambda x: b*x,
            x0,
            tmax,
            exact_solution=lambda x0,t,w:x0*np.exp((a-0.5*b*b)*t+b*w)
        )

        solver = pychastic.sde_solver.SDESolver(adaptive=False,scheme='euler',dt=2**(-4))

        n_batches = 20
        n_per_batch = 100
        results = solver.solve_many(problem=problem,n_trajectories=n_batches*n_per_batch).reshape((n_batches,n_per_batch))

        s = results.shape
        results_flat = results.flatten()
        def f(sol):
            x = sol['solution_values'][-1]
            x_exact = problem.exact_solution(problem.x0, sol['time_values'][-1], sol['wiener_values'][-1])
            assert np.isclose(sol['time_values'][-1], problem.tmax)
            e = abs(x - x_exact)
            return e
        errors = np.array([f(x) for x in results_flat]).reshape(s)
        
        print(errors)

        #self.assertAlmostEqual( var , 1 , delta=5.0/np.sqrt(T), 
        #       msg = f'Variance of Wiener increments incorrect: {var}')
        
if __name__ == '__main__':
    #np.random.seed(0)
    unittest.main()
