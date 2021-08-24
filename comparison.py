from pychastic.problems.kp_4_27 import KloedenPlaten4_27
from pychastic.sde_solver import SDESolver
import pychastic
import matplotlib.pyplot as plt
import numpy as np


problem = KloedenPlaten4_27

def solvers():
    dts = [0.01, 0.001]#, 0.0001]
    rhos = [0.1, 0.01]
    
    for dt in dts:
        yield SDESolver(adaptive=False, scheme='euler', dt=dt), {'name': f'Euler, dt={dt}', 'dt': dt, 'e_terms': 0, 'adaptive': False, 'scheme': 'euler'}
        yield SDESolver(adaptive=False, scheme='milstein', dt=dt), {'name': f'Milstein, dt={dt}', 'dt': dt, 'e_terms': 0, 'adaptive': False, 'scheme': 'milstein'}

        for rho in rhos:         
            yield SDESolver(adaptive=True, scheme='euler', dt=dt, target_mse_density=rho, error_terms=1), {'name': f'Euler ad, dt={dt}, eterms=1, rho={rho}', 'dt': dt, 'e_terms': 1, 'rho': rho, 'adaptive': True, 'scheme': 'euler'}
            yield SDESolver(adaptive=True, scheme='euler', dt=dt, target_mse_density=rho, error_terms=2), {'name': f'Euler ad, dt={dt}, eterms=2, rho={rho}', 'dt': dt, 'e_terms': 2, 'rho': rho, 'adaptive': True, 'scheme': 'euler'}
            yield SDESolver(adaptive=True, scheme='milstein', dt=dt, target_mse_density=rho, error_terms=1), {'name': f'Milstein ad, dt={dt}, eterms=1, rho={rho}', 'dt': dt, 'e_terms': 1, 'rho': rho, 'adaptive': True, 'scheme': 'milstein'}

import pandas as pd
from tqdm import tqdm
results = []

data = []
n_repetitions = 2
wieners = [pychastic.wiener.Wiener() for i in range(n_repetitions)]
for solver, solver_params in tqdm(list(solvers())):
    solutions = solver.solve_many(problem, wieners)
    for solution, wiener in zip(solutions, wieners):
        t_end = solution['time_values'][-1]
        assert np.isclose(t_end, problem.tmax)
        end_value = solution['solution_values'][-1]
        error = problem.exact_solution(problem.x0, problem.tmax, wiener.get_w(problem.tmax))
        data.append(dict(
            error=error,
            mean_dt=problem.tmax / (len(solution['time_values'])-1),
            **solver_params
        ))

   
data = pd.DataFrame(data)
print(data)
data.to_csv('data.csv')
exit()

t = np.linspace(0, problem.tmax, 150)
w_ = np.array([w.get_w(t_) for t_ in t])
x = problem.exact_solution(problem.x0, t, w_)
plt.plot(t, x, c='black', label='exact')

plt.legend()
plt.savefig('comparison.png')
