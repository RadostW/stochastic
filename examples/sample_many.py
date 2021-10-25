from pychastic.problems.kp_4_27 import KloedenPlaten4_27
from pychastic.sde_solver import SDESolver
import jax.numpy as jnp

problem = KloedenPlaten4_27 # some scalar problem
problem.x0 = jnp.array([1.0])
problem.tmax = 50.0
solver = SDESolver()
solver.dt = 0.01
solution = solver.solve_many(problem,1000)

import matplotlib.pyplot as plt

mean = jnp.mean(solution['solution_values'],axis=0).squeeze()
sigma = jnp.std(solution['solution_values'],axis=0).squeeze()
time = solution['time_values'][0].squeeze()
example_trajectories = solution['solution_values'][:100].squeeze()

fig, axs = plt.subplots(2)

axs[0].violinplot(solution['solution_values'][:,::500].squeeze().transpose(),positions = time[::500],widths=4.0)
axs[1].plot(time,mean, color = 'k')
axs[1].plot(time[::100],example_trajectories.transpose()[::100,:], color = 'b', alpha = 0.1)
plt.show()

#plt.plot(solution['time_values'], solution['solution_values']-exact)
#plt.title('Error')
#plt.savefig('error.png')
#plt.close()
