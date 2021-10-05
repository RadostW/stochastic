from pychastic.problems.kp_4_27 import KloedenPlaten4_27
from pychastic.sde_solver import SDESolver
import pychastic
import matplotlib.pyplot as plt
import numpy as np


problem = KloedenPlaten4_27
problem.tmax = 10

def solvers():
    dts = np.exp(np.linspace(np.log(0.001), np.log(0.1), 10))
    rhos = list()#[0.1, 0.01]
    
    for dt in dts:
        yield SDESolver(adaptive=False, scheme='euler', dt=dt), {'name': f'Euler', 'dt': dt, 'e_terms': 0, 'adaptive': False, 'scheme': 'euler'}
        yield SDESolver(adaptive=False, scheme='milstein', dt=dt), {'name': f'Milstein', 'dt': dt, 'e_terms': 0, 'adaptive': False, 'scheme': 'milstein'}
        continue
        for rho in rhos:         
            yield SDESolver(adaptive=True, scheme='euler', dt=dt, target_mse_density=rho, error_terms=1), {'name': f'Euler ad, eterms=1, rho={rho}', 'dt': dt, 'e_terms': 1, 'rho': rho, 'adaptive': True, 'scheme': 'euler'}
            yield SDESolver(adaptive=True, scheme='euler', dt=dt, target_mse_density=rho, error_terms=2), {'name': f'Euler ad, eterms=2, rho={rho}', 'dt': dt, 'e_terms': 2, 'rho': rho, 'adaptive': True, 'scheme': 'euler'}
            yield SDESolver(adaptive=True, scheme='milstein', dt=dt, target_mse_density=rho, error_terms=1), {'name': f'Milstein ad, eterms=1, rho={rho}', 'dt': dt, 'e_terms': 1, 'rho': rho, 'adaptive': True, 'scheme': 'milstein'}

solvers = list(solvers())
import pandas as pd
from tqdm import tqdm
results = []

file = open('trajs.txt', 'w')

data = []
n_repetitions = 50
wieners = [pychastic.wiener.Wiener(seed=i) for i in range(n_repetitions)]
for solver, solver_params in solvers:
    del solver_params['dt']
    solutions = solver.solve_many(problem, wieners)
    for solution, wiener in zip(solutions, wieners):
        
        t_end = solution['time_values'][-1]
        assert np.isclose(t_end, problem.tmax)        
        end_value = solution['solution_values'][-1]

        error = problem.exact_solution(problem.x0, problem.tmax, wiener.get_w(problem.tmax)) - end_value
        mean_dt = problem.tmax / (len(solution['solution_values'])-1)
        data.append(dict(
            error=error,
            mean_dt=mean_dt,
            **solver_params
        ))
        np.savetxt(file, solution['solution_values'], newline=' ')
        file.write('\n')
#file.close()
  
data = pd.DataFrame(data)
data.to_csv('data.csv')
exit()

t = np.linspace(0, problem.tmax, 150)
w_ = np.array([w.get_w(t_) for t_ in t])
x = problem.exact_solution(problem.x0, t, w_)
plt.plot(t, x, c='black', label='exact')

plt.legend()
plt.savefig('comparison.png')
