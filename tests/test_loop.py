import pychastic                      # solving sde
import pygrpy.jax_grpy_tensors        # hydrodynamic interactions
import pywrithe                       # computing writhe of closed curve
import jax.numpy as jnp               # jax array operations
import jax                            # taking gradients
import matplotlib.pyplot as plt       # plotting
import numpy as np                    # post processing trajectory
import math as ma                     # math.pi
from tqdm import tqdm                 # progess bar

from jax.config import config         # extra debug
config.update("jax_debug_nans",True)  # throw on nans


'''
Model paramters to be set for simulation
'''
beam_length = 1.0                     # length of the filament
beam_diameter = 28.4 / 1111.          # filament diameter
n_beads = 40                          # number of beads modelling the filament
kbT = 1.0                             # fluctuation energy
persistence_length = 500.0 / 1111.    # L_p = EI / kbT (500 angstroms)
stretch_length = 0.037 / 1111.        # L_s = kbT / EA (0.037 angstroms)
linking_number = 3.5                  # linking number of loop


'''
Resulting parameters computed from the above
'''
equilibirum_distance = beam_length / n_beads   # equilibirum for hookean springs
spring_constant =                              # stiffness of elongation springs
bending_constant = 
twisting_constant = 
overlap_constant = 70.0 * kbT                  # fixed value

radii = jnp.array([1.0 for x in range(n_beads)])

relative_temperature = 0.3

def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(n_beads,3))
     distances = jnp.sum((locations-jnp.roll(locations,1,axis=0))**(2.0),axis=1)**(0.5)
     extension_ene = spring_constant*jnp.sum((distances - equilibirum_distance)**2)

     curvatures = jnp.sum((locations-2*jnp.roll(locations,1,axis=0)+jnp.roll(locations,2,axis=0))**(2.0),axis=1)**(0.5)
     bend_ene = bending_constant*jnp.sum(curvatures**2)

     twist_ene = twisting_constant * (linking_number - pywrithe.writhe_jax(locations))**2

     overlap_ene = overlap_constant * jnp.sum(jnp.tanh(1.5-jnp.sum((locations[:,jnp.newaxis,:] - locations[jnp.newaxis,:,:])**2,axis=-1))+1.0)

     return extension_ene + bend_ene + twist_ene + overlap_ene

def drift(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     force = -jax.grad(u_ene)(x)
     return jnp.matmul(mu,force)

def noise(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     return relative_temperature*jnp.sqrt(2)*jnp.linalg.cholesky(mu)

problem = pychastic.sde_problem.VectorSDEProblem(
      drift,
      noise,
      x0 = jnp.reshape(jnp.array([[1.5*(n_beads/ma.pi)*ma.cos(2.0*ma.pi*x/n_beads),0.8*(n_beads/ma.pi)*ma.sin(2.0*ma.pi*x/n_beads),0.01*ma.sin(2.5*ma.pi*x/n_beads)] for x in range(n_beads)]),(3*n_beads,)), 
      dimension = 3*n_beads,
      noiseterms = 3*n_beads,
      tmax = 5000.0)

solver = pychastic.sde_solver.VectorSDESolver(dt = 0.2)
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

