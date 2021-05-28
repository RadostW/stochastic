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
    tmax = 50,
)

w = pychastic.wiener.VectorWienerWithI(2)
variant_a = []
t0 = 0
while t0 < problem.tmax:
    t0 += 0.01
    variant_a.append(t0)
t0 = 0
while t0 < problem.tmax:
    t0 += 0.1
    variant_a.append(t0)

for t in sorted(variant_a):
    w.get_w(t)

solver = pychastic.sde_solver.VectorSDESolver()
solver.scheme = "euler"
solver.dt = 0.01
eul_better_sol = solver.solve(problem,w)

solver.dt = 0.1
solver.scheme = "milstein"
mil_sol = solver.solve(problem,w)
solver.scheme = "euler"
eul_sol = solver.solve(problem,w)

plt.plot(mil_sol['time_values'],mil_sol['solution_values'][:,0],label='milstein')
plt.plot(eul_sol['time_values'],eul_sol['solution_values'][:,0],label='euler')
plt.plot(eul_better_sol['time_values'],eul_better_sol['solution_values'][:,0],label='euler_x10')
plt.legend()
plt.show()