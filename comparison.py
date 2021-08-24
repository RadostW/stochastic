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
wieners = [pychastic.wiener.Wiener() for i in range(n_repetitions)]
for solver, solver_params in tqdm(list(solvers())):
    solver.solve_many(problem, wieners)
    
exit()
results = pd.DataFrame(results)
print(results)
results.to_csv('results.csv')

t = np.linspace(0, problem.tmax, 150)
w_ = np.array([w.get_w(t_) for t_ in t])
x = problem.exact_solution(problem.x0, t, w_)
plt.plot(t, x, c='black', label='exact')

plt.legend()
plt.savefig('comparison.png')
