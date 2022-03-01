# Following code implements simuation of motion of
# two hydrodynamically interacting brownian spheres conneced
# with a harmonic spring

import pychastic                   # solving sde
import pygrpy.jax_grpy_tensors     # hydrodynamic interactions
import jax.numpy as jnp            # jax array operations
import jax                         # taking gradients
import matplotlib.pyplot as plt    # plotting

radii = jnp.array([1.0,0.2]) # sizes of spheres we're using
def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(2,3))
     distance = jnp.sqrt(jnp.sum((locations[0] - locations[1])**2))
     return (distance-4.0)**2

def drift(x):
     locations = jnp.reshape(x,(2,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     force = -jax.grad(u_ene)(x)
     return jnp.matmul(mu,force)

def noise(x):
     locations = jnp.reshape(x,(2,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     return jnp.sqrt(2)*jnp.linalg.cholesky(mu)

problem = pychastic.sde_problem.SDEProblem(
      drift,
      noise,
      x0 = jnp.reshape(jnp.array([[0.,0.,0.],[0.,0.,4.]]),(6,)),
      # dimension = 6,
      # noiseterms = 6,
      tmax = 500.0)
solver = pychastic.sde_solver.SDESolver()

trajectory = solver.solve(problem) # takes about 5 seconds

plt.plot(trajectory['time_values'],trajectory['solution_values'][:,0])
plt.plot(trajectory['time_values'],trajectory['solution_values'][:,3])
plt.show()
