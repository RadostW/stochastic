import pychastic
import numpy as np
import time

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

solver = pychastic.sde_solver.SDESolver(adaptive=False,scheme='euler',dt=2**(-6))


start = time.time()
results = np.array(solver.solve_many(problem=problem,n_trajectories=4000))
print(f'Time: {time.time()-start}')
