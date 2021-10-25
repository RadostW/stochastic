import pychastic                      # solving sde
import pygrpy.jax_grpy_tensors        # hydrodynamic interactions
import pywrithe                       # computing writhe of closed curve
import jax.numpy as jnp               # jax array operations
import jax                            # taking gradients
import matplotlib.pyplot as plt       # plotting
import numpy as np                    # post processing trajectory
import math as ma                     # math.pi
from tqdm import tqdm                 # progess bar

#from jax.config import config         # extra debug
#config.update("jax_debug_nans",True)  # throw on nans


'''
Model paramters to be set for simulation
'''
beam_length = 1.0                          # length of the filament
beam_hydrodynamic_diameter = 28.4 / 1111.  # filament diameter (hydrodynamic)
beam_steric_diameter = 20.0 / 1111.        # filament diameter (steric)
n_beads = 40                               # number of beads modelling the filament
kbT = 1.0                                  # fluctuation energy
persistence_length = 500.0 / 1111.         # L_p = EI / kbT (500 angstroms)
#stretch_length = 0.037 / 1111.            # L_s = kbT / EA (0.037 angstroms)
stretch_length = 3.0 / 1111.               # CHANGED VALUE
linking_number = 5.5                       # linking number of loop


'''
Resulting parameters computed from the above
'''
equilibirum_distance = 0.5 * beam_length / n_beads                              # equilibirum of elongation springs
spring_constant = kbT / (stretch_length * equilibirum_distance)                 # stiffness of elongation springs

bending_constant = kbT * persistence_length * (1.0/4.0) * equilibirum_distance  # stiffness of bending springs
twisting_constant = kbT * persistence_length * (2.0/3.0) * beam_length          # stiffness of twisting springs

overlap_constant = 70.0 * kbT                                                   # stiffness of overlap springs
overlap_distance = 0.6 * beam_steric_diameter                                   # smear width of overlap springs

radii = jnp.array([(beam_hydrodynamic_diameter / 2.0) for x in range(n_beads)]) # hydrodynamic radii of beads

near_mask = jnp.array(np.ones((n_beads,n_beads))-np.eye(n_beads)-np.roll(np.eye(n_beads),1,axis=0)-np.roll(np.eye(n_beads),1,axis=0)) # zero for |i-j|<=1

def loop_writhe(x):
     locations = jnp.reshape(x,(n_beads,3))
     return pywrithe.writhe_jax(locations)

def u_ene(x): # potential energy shape
     locations = jnp.reshape(x,(n_beads,3))
     distances = jnp.sum((locations-jnp.roll(locations,1,axis=0))**(2.0),axis=1)**(0.5)
     extension_ene = spring_constant*0.5*jnp.sum((distances - equilibirum_distance)**2)

     curvatures = jnp.sum((locations-2*jnp.roll(locations,1,axis=0)+jnp.roll(locations,2,axis=0))**(2.0),axis=1)**(0.5) / (equilibirum_distance**2)
     bend_ene = bending_constant*0.5*jnp.sum(curvatures**2)

     twist_ene = twisting_constant * 0.5 * (4.0*ma.pi*ma.pi) * (linking_number - pywrithe.writhe_jax(locations))**2

     overlap_ene = overlap_constant * (0.5*jnp.sum(jnp.tanh(
                                                (beam_steric_diameter**2-near_mask*jnp.sum((locations[:,jnp.newaxis,:] - locations[jnp.newaxis,:,:])**2,axis=-1))/(overlap_distance**2)
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
     return jnp.sqrt(2*kbT)*jnp.linalg.cholesky(mu)

#problem = pychastic.sde_problem.SDEProblem(
#      drift,
#      noise,
#      x0 = (beam_length / (2.0*ma.pi))*jnp.reshape(jnp.array([
#                        [1.1*ma.cos(2.0*ma.pi*x/n_beads),0.9*ma.sin(2.0*ma.pi*x/n_beads),0.05*ma.sin(4.0*ma.pi*x/n_beads)] for x in range(n_beads)
#                      ]),(3*n_beads,)), 
#      #dimension = 3*n_beads,
#      #noiseterms = 3*n_beads,
#      tmax = 6000.0*10.0**(-6.0))

#solver = pychastic.sde_solver.SDESolver(dt = 0.05*10.0**(-6.0))
#solutions = solver.solve_many(problem,1)


#
# plotting
#

trajectory = np.loadtxt('data_loop.csv', delimiter = ',')
writhes = jax.lax.map(loop_writhe,trajectory)
plt.plot(writhes)

#trajectory = trajectories[0]
#trajectory = {'solution_values' : np.loadtxt('data_loop.csv', delimiter = ','), 'time_values' : np.arange(0,tmax,0.05*10.0**(-6.0))}
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,0])
#plt.plot(trajectory['time_values'],trajectory['solution_values'][:,3])


#plt.plot(trajectory['solution_values'][:,0],trajectory['solution_values'][:,1])
#plt.plot(trajectory['solution_values'][:,9],trajectory['solution_values'][:,10])
#np.savetxt('data_loop.csv', trajectory['solution_values'], delimiter=',')

#sol =  np.array([x['solution_values'] for x in trajectories]);
#plt.plot(trajectories[0]['time_values'],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,0:3]-sol[:,0,np.newaxis,0:3])**2,axis=2),axis=0),label='First, big bead') # big bead
#plt.plot(trajectories[0]['time_values'],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,9:12]-sol[:,0,np.newaxis,9:12])**2,axis=2),axis=0),label='Last, small bead') # small bead
#plt.plot([0.0,problem.tmax],[0.0,0.3333*(1.0/ma.pi)*problem.tmax]) # largest bead -- theory
#plt.plot([0.0,problem.tmax],[0.0,0.2898*(1.0/ma.pi)*problem.tmax]) # BD sim Cichocki et al -- theory
#plt.xlabel(r"Dimensionless time ($t/\tau$)")
#plt.ylabel(r"Mean square displacement ($\mathbb{E}|q|^2$)")
#plt.legend()

plt.show()

