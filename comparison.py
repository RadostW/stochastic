import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['USE_SIMPLE_THREADED_LEVEL3']='1'

from pychastic.problems.kp_4_27 import KloedenPlaten4_27
import pychastic
import matplotlib.pyplot as plt
import numpy as np


problem = KloedenPlaten4_27

def solvers():
    solver = pychastic.sde_solver.SDESolver()
    dts = [0.01, 0.001, 0.0001]
    rhos = [0.1, 0.01, 0.001]
    
    

    for dt in dts:
        solver.dt = dt
        solver.scheme = 'euler'
        solver.adaptive = False
        yield solver, {'name': 'Euler', 'dt': dt, 'e_terms': 0}

        solver.scheme = 'milstein'
        solver.adaptive = False
        yield solver, {'name': 'Milstein', 'dt': dt, 'e_terms': 0}

        for rho in rhos:            
            solver.scheme = 'euler'
            solver.adaptive = True
            solver.error_terms = 1
            solver.target_mse_density = rho
            yield solver, {'name': 'Euler ad', 'dt': dt, 'e_terms': 1, 'rho': rho}
            
            solver.scheme = 'euler'
            solver.adaptive = True
            solver.error_terms = 2
            solver.target_mse_density = rho
            yield solver, {'name': 'Euler ad', 'dt': dt, 'e_terms': 2, 'rho': rho}
            
            solver.scheme = 'milstein'
            solver.adaptive = True
            solver.error_terms = 1
            solver.target_mse_density = rho
            yield solver, {'name': 'Milstein ad', 'dt': dt, 'e_terms': 1, 'rho': rho}

import pandas as pd
from tqdm import tqdm
results = []

n_repetitions = 6
wieners = [pychastic.wiener.Wiener(seed=i) for i in range(n_repetitions)]
for n_repetition in tqdm(list(range(n_repetitions))):
    # reset underlying wiener
    w = pychastic.wiener.Wiener(seed=n_repetition)
    end_value = problem.exact_solution(problem.x0, problem.tmax, w.get_w(problem.tmax))
    for solver, solver_params in solvers():
        result = solver.solve(problem, w)
        assert np.isclose(result['time_values'][-1], problem.tmax, rtol=1e-4)
        results.append(dict(
            end_error = end_value - result['solution_values'][-1],
            steps = len(result['solution_values']) - 1,
            main_loop_time_ms = result['main_loop_time_ms'],
            n_repetition=n_repetition,
            **solver_params
        ))
        #plt.plot(result['time_values'], result['solution_values'], label=solver_params['name'])#, marker='x')

results = pd.DataFrame(results)
print(results)
results.to_csv('results.csv')

exit()
t = np.linspace(0, problem.tmax, 150)
w_ = np.array([w.get_w(t_) for t_ in t])
x = problem.exact_solution(problem.x0, t, w_)
plt.plot(t, x, c='black', label='exact')

plt.legend()
plt.savefig('comparison.png')
