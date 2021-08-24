import pychastic                   # solving sde
import pygrpy.jax_grpy_tensors     # hydrodynamic interactions
import jax.numpy as jnp            # jax array operations
import jax                         # taking gradients
import matplotlib.pyplot as plt    # plotting
import numpy as np                 # post processing trajectory
radii = jnp.array([3.0,1.0,1.0,1.0]) # sizes of spheres we're using
n_beads = len(radii)
equilibrium_dist = 4.0
spring_constant = 5.5
def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(n_beads,3))
     distance_ab = jnp.sqrt(jnp.sum((locations[0] - locations[1])**2))
     distance_bc = jnp.sqrt(jnp.sum((locations[1] - locations[2])**2))
     distance_cd = jnp.sqrt(jnp.sum((locations[2] - locations[3])**2))
     ene = (
           spring_constant*(distance_ab-equilibrium_dist)**2
         + spring_constant*(distance_bc-equilibrium_dist)**2
         + spring_constant*(distance_cd-equilibrium_dist)**2
            )
     return ene

def drift(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     force = -jax.grad(u_ene)(x)
     return jnp.matmul(mu,force)

def noise(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     return jnp.sqrt(2)*jnp.linalg.cholesky(mu)

problem = pychastic.sde_problem.VectorSDEProblem(
      drift,
      noise,
      #x0 = jnp.reshape(jnp.array([[0.,0.,0.],[0.,0.,4.]]),(3*n_beads,)), # two beads
      x0 = jnp.reshape(jnp.array([[-2.,0.,0.],[2.,0.,0.],[6.,0.,0.],[10.,0.,0.]]),(3*n_beads,)), # four beads
      dimension = 3*n_beads,
      noiseterms = 3*n_beads,
      tmax = 10000.0)

solver = pychastic.sde_solver.VectorSDESolver(dt = 0.1)
trajectories = np.array([solver.solve(problem) for x in range(2)])


#
# plotting
#

#trajectory = trajectories[0]
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,0])
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,3])

trajectory = trajectories[0]
plt.plot(trajectory['solution_values'][:,0],trajectory['solution_values'][:,1])
plt.plot(trajectory['solution_values'][:,9],trajectory['solution_values'][:,10])
np.savetxt('data.csv', trajectory['solution_values'], delimiter=',')

#sol =  np.array([x['solution_values'] for x in trajectories]);
#plt.plot(trajectories[0]['time_values'],np.sum(np.sum((sol[:,:,0:3]-sol[:,0,np.newaxis,0:3])**2,axis=2),axis=0)) # big bead
#plt.plot(trajectories[0]['time_values'],np.sum(np.sum((sol[:,:,9:12]-sol[:,0,np.newaxis,9:12])**2,axis=2),axis=0)) # small bead

plt.show()

