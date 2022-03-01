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
beam_length = 1.0                          # length of the filament
beam_hydrodynamic_diameter = 28.4 / 1111.  # filament diameter (hydrodynamic)
beam_steric_diameter = 20.0 / 1111.        # filament diameter (steric)
n_beads = 40                               # number of beads modelling the filament
kbT = 1.0                                  # fluctuation energy
persistence_length = 500.0 / 1111.         # L_p = EI / kbT (500 angstroms)
stretch_length = 0.037 / 1111.             # L_s = kbT / EA (0.037 angstroms)
linking_number = 2.7                       # linking number of loop


'''
Resulting parameters computed from the above
'''
equilibirum_distance = beam_length / n_beads                                    # equilibirum of elongation springs
spring_constant = kbT / (stretch_length * equilibirum_distance)                 # stiffness of elongation springs

bending_constant = kbT * persistence_length * equilibirum_distance              # stiffness of bending springs
twisting_constant = kbT * persistence_length * (2.0/3.0) * beam_length          # stiffness of twisting springs

overlap_constant = 70.0 * kbT                                                   # stiffness of overlap springs
overlap_distance = 0.6 * beam_steric_diameter                                   # smear width of overlap springs

radii = jnp.array([(beam_hydrodynamic_diameter / 2.0) for x in range(n_beads)]) # hydrodynamic radii of beads

def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(n_beads,3))
     distances = jnp.sum((locations-jnp.roll(locations,1,axis=0))**(2.0),axis=1)**(0.5)
     extension_ene = spring_constant*0.5*jnp.sum((distances - equilibirum_distance)**2)

     curvatures = jnp.sum((locations-2*jnp.roll(locations,1,axis=0)+jnp.roll(locations,2,axis=0))**(2.0),axis=1)**(0.5) / (equilibirum_distance**2)
     bend_ene = bending_constant*0.5*jnp.sum(curvatures**2)

     twist_ene = twisting_constant * 0.5 * (4.0*ma.pi*ma.pi) * (linking_number - pywrithe.writhe_jax(locations))**2

     overlap_ene = overlap_constant * (0.5*jnp.sum(jnp.tanh(
                                                (beam_steric_diameter**2-jnp.sum((locations[:,jnp.newaxis,:] - locations[jnp.newaxis,:,:])**2,axis=-1))/(overlap_distance**2)
                                                )+1.0)-n_beads)

     return extension_ene + bend_ene + twist_ene + overlap_ene

def drift(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     force = -jax.grad(u_ene)(x)
     return jnp.matmul(mu,force)

def noise(x):
     locations = jnp.reshape(x,(n_beads,3))
     mu = pygrpy.jax_grpy_tensors.muTT(locations,radii)
     return jnp.sqrt(2)*jnp.linalg.cholesky(mu)

problem = pychastic.sde_problem.SDEProblem(
      drift,
      noise,
      x0 = (beam_length / (2.0*ma.pi))*jnp.reshape(jnp.array([
                        [ma.cos(2.0*ma.pi*x/n_beads),ma.sin(2.0*ma.pi*x/n_beads),0.0] for x in range(n_beads)
                      ]),(3*n_beads,)),
      # dimension = 3*n_beads,
      # noiseterms = 3*n_beads,
      tmax = 6000.0*10.0**(-6.0))

solver = pychastic.sde_solver.SDESolver(dt = 0.1*10.0**(-6.0))
trajectories = np.array([solver.solve(problem) for x in tqdm(range(1))])

#
# plotting
#

trajectory = trajectories[0]
plt.plot(trajectory['solution_values'][:,0],trajectory['solution_values'][:,1])
plt.plot(trajectory['solution_values'][:,9],trajectory['solution_values'][:,10])
np.savetxt('data_loop.csv', trajectory['solution_values'], delimiter=',')

plt.show()

