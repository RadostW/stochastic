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

        n_batches = 10
        n_per_batch = 100
        wieners = [pychastic.wiener.Wiener(seed=i) for i in range(n_batches*n_per_batch)]
        results = np.array(solver.solve_many(problem=problem,wieners=wieners)).reshape((n_batches,n_per_batch))

        s = results.shape
        results_flat = results.flatten()
        def f(sol):
            x = sol['solution_values'][-1]
            x_exact = problem.exact_solution(problem.x0, sol['time_values'][-1], sol['wiener_values'][-1])
            assert np.isclose(sol['time_values'][-1], problem.tmax)
            e = abs(x - x_exact)
            return e
        errors = np.array([f(x) for x in results_flat]).reshape(s)
        
        batch_error_estimates = errors.mean(axis=-1)

        mean_error_estimate = batch_error_estimates.mean()
        mean_error_confidence = 2*batch_error_estimates.std() / np.sqrt(n_batches)

        # Kloden-Platen Figure 9.3.1
        self.assertAlmostEqual(mean_error_estimate,0.2870,delta=0.1, msg='Euler-Maruyama error estimate inconsistent with Kloden-Platen')
        self.assertAlmostEqual(mean_error_confidence,0.004914,delta=0.002, msg='Euler-Maruyama error confidence inconsistent with Kloden-Platen')

class TestMilstein(unittest.TestCase):
    def test_milstein_error_size(self):
        
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

        solver = pychastic.sde_solver.SDESolver(adaptive=False,scheme='milstein',dt=2**(-6))

        n_batches = 20
        n_per_batch = 100
        wieners = [pychastic.wiener.Wiener(seed=i) for i in range(n_batches*n_per_batch)]
        results = np.array(solver.solve_many(problem=problem,wieners=wieners)).reshape((n_batches,n_per_batch))

        s = results.shape
        results_flat = results.flatten()
        def f(sol):
            x = sol['solution_values'][-1]
            x_exact = problem.exact_solution(problem.x0, sol['time_values'][-1], sol['wiener_values'][-1])
            assert np.isclose(sol['time_values'][-1], problem.tmax)
            e = abs(x - x_exact)
            return e
        errors = np.array([f(x) for x in results_flat]).reshape(s)
        
        batch_error_estimates = errors.mean(axis=-1)

        mean_error_estimate = batch_error_estimates.mean()
        #mean_error_confidence = 2*batch_error_estimates.std() / np.sqrt(n_batches)

        # Kloden-Platen Figure 10.3.1
        # 2^(-3.7058) = 0.0766378
        self.assertAlmostEqual(mean_error_estimate,0.0766378, delta=0.01, msg='Milstein error estimate inconsistent with Kloden-Platen')
        #self.assertAlmostEqual(mean_error_confidence,0.004914,delta=0.002, msg='Milstein error confidence inconsistent with Kloden-Platen')


class TestWagnerPlaten(unittest.TestCase):
    def test_wagner_platen_error_size(self):
        
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

        solver = pychastic.sde_solver.SDESolver(adaptive=False,scheme='wagner_platen',dt=2**(-6))

        n_batches = 20
        n_per_batch = 100
        wieners = [pychastic.wiener.WienerWithZ(seed=i) for i in range(n_batches*n_per_batch)]
        results = np.array(solver.solve_many(problem=problem,wieners=wieners)).reshape((n_batches,n_per_batch))

        s = results.shape
        results_flat = results.flatten()
        def f(sol):
            x = sol['solution_values'][-1]
            x_exact = problem.exact_solution(problem.x0, sol['time_values'][-1], sol['wiener_values'][-1])
            assert np.isclose(sol['time_values'][-1], problem.tmax)
            e = abs(x - x_exact)
            return e
        errors = np.array([f(x) for x in results_flat]).reshape(s)
        
        batch_error_estimates = errors.mean(axis=-1)

        mean_error_estimate = batch_error_estimates.mean()
        #mean_error_confidence = 2*batch_error_estimates.std() / np.sqrt(n_batches)

        # Kloden-Platen Figure 10.3.1
        # 2^(-10.66) = 0.000618045
        self.assertAlmostEqual(mean_error_estimate,0.000618045, delta=0.0001, msg='Wagner-Platen error estimate inconsistent with Kloden-Platen')
        

        
if __name__ == '__main__':
    #np.random.seed(0)
    unittest.main()
