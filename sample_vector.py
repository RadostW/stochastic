import pychastic
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# problem = pychastic.sde_problem.VectorSDEProblem(
#     lambda x: jnp.array([x[0]**3 - 3*x[0]*x[1]**2, -x[1]**3 + 3*x[1]*x[0]**2]),
#     lambda x: jnp.array([[x[0]**2 - x[1]**2, 2*x[0]*x[1]]]).T,
#     2,
#     1,
#     jnp.array([1.5, 0.5]),
#     1,
# )

problem = pychastic.sde_problem.VectorSDEProblem(
    lambda x: jnp.array([jnp.sin(x[0]),2.,3.]),
    lambda x: jnp.array([[1. * x[0],0.],[0.,2. * x[1]],[3. * x[2],0.]]),
    dimension = 3,
    noiseterms = 2,
    x0 = jnp.array([1.5, 0.5,1.1]),
    tmax = 1,
)

solver = pychastic.sde_solver.VectorSDESolver()
solver.scheme = "euler"
solver.adaptive = True
solver.dt = 0.01
solver.target_mse_density = 1
eul_sol = solver.solve(problem)

plt.plot(eul_sol['time_values'],eul_sol['solution_values'][:,0],label='euler', marker='x')
plt.legend()
plt.savefig('muldim_adaptuve.png')