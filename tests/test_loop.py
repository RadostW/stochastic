import pychastic                   # solving sde
import pygrpy.jax_grpy_tensors     # hydrodynamic interactions
import pywrithe                    # computing writhe of closed curve
import jax.numpy as jnp            # jax array operations
import jax                         # taking gradients
import matplotlib.pyplot as plt    # plotting
import numpy as np                 # post processing trajectory
import math as ma                  # math.pi
from tqdm import tqdm              # progess bar

radii = jnp.array([1.0 for x in range(40)]) # sizes of spheres we're using
n_beads = len(radii)
equilibrium_dist = 2.0
spring_constant = 5.5
bending_constant = 1.7 # 70 / (4 pi^2) (correct?) value for 40 beads and persistency = length
twisting_constant = 0.1
linking_number = 0.0

def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(n_beads,3))
     distances = jnp.sum((locations[1:]-locations[:-1])**(2.0),axis=1)**(0.5)
     extension_ene = spring_constant*jnp.sum((distances - equilibrium_dist)**2)

     curvatures = jnp.sum((locations[2:]-2*locations[1:-1]+locations[:-2])**(2.0),axis=1)**(0.5)
     bend_ene = bending_constant*jnp.sum(curvatures**2)

     glue_ene = (
          spring_constant*(jnp.sum((locations[-1]-locations[0])**2)**0.5-equilibrium_dist)**2
        +
          bending_constant*(
            jnp.sum((locations[-2]-2*locations[-1]+locations[0])**2)**0.5
          + jnp.sum((locations[-1]-2*locations[0]+locations[1])**2)**0.5
          )**2
        )

     twist_ene = twisting_constant*(linking_number - pywrithe.writhe_jax(locations))**2

     return extension_ene + bend_ene + glue_ene + twist_ene

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
      x0 = jnp.reshape(jnp.array([[(n_beads/ma.pi)*ma.cos(2.0*ma.pi*x/n_beads),(n_beads/ma.pi)*ma.sin(2.0*ma.pi*x/n_beads),0.01*ma.sin(2.5*ma.pi*x/n_beads)] for x in range(n_beads)]),(3*n_beads,)), # four beads
      dimension = 3*n_beads,
      noiseterms = 3*n_beads,
      tmax = 1.0)

solver = pychastic.sde_solver.VectorSDESolver(dt = 0.1)
trajectories = np.array([solver.solve(problem) for x in tqdm(range(1))])


#
# plotting
#

#trajectory = trajectories[0]
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,0])
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,3])

trajectory = trajectories[0]
plt.plot(trajectory['solution_values'][:,0],trajectory['solution_values'][:,1])
plt.plot(trajectory['solution_values'][:,9],trajectory['solution_values'][:,10])
np.savetxt('data_loop.csv', trajectory['solution_values'], delimiter=',')

#sol =  np.array([x['solution_values'] for x in trajectories]);
#plt.plot(trajectories[0]['time_values'],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,0:3]-sol[:,0,np.newaxis,0:3])**2,axis=2),axis=0),label='First, big bead') # big bead
#plt.plot(trajectories[0]['time_values'],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,9:12]-sol[:,0,np.newaxis,9:12])**2,axis=2),axis=0),label='Last, small bead') # small bead
#plt.plot([0.0,problem.tmax],[0.0,0.3333*(1.0/ma.pi)*problem.tmax]) # largest bead -- theory
#plt.plot([0.0,problem.tmax],[0.0,0.2898*(1.0/ma.pi)*problem.tmax]) # BD sim Cichocki et al -- theory
#plt.xlabel(r"Dimensionless time ($t/\tau$)")
#plt.ylabel(r"Mean square displacement ($\mathbb{E}|q|^2$)")
#plt.legend()

plt.show()

