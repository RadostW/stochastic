# Following code is implementation of simulation published in:
# "Diffusion coefficients of elastic macromolecules"
# B. Cichocki, M. Rubin, A. Niedzwiecka & P. Szymczak
# J. Fluid Mech. (2019)
# doi:10.1017/jfm.2019.652

import pychastic                   # solving sde
import pygrpy.jax_grpy_tensors     # hydrodynamic interactions
import jax.numpy as jnp            # jax array operations
import jax                         # taking gradients
import matplotlib.pyplot as plt    # plotting
import numpy as np                 # post processing trajectory
import math as ma                  # math.pi
from tqdm import tqdm              # progess bar

radii = jnp.array([3.0,1.0,1.0,1.0]) # sizes of spheres we're using
#radii = jnp.array([1.0]) # SINGLE BEAD BENCHMARK
n_beads = len(radii)
equilibrium_dist = 4.0
spring_constant = 5.5

def u_ene(x): # potential energy shape
     #return 0.0 # SINGLE BEAD BENCHMARK
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

problem = pychastic.sde_problem.SDEProblem(
      drift,
      noise,
      #x0 = jnp.reshape(jnp.array([[0.,0.,0.],[0.,0.,4.]]),(3*n_beads,)), # two beads
      x0 = jnp.reshape(jnp.array([[-2.,0.,0.],[2.,0.,0.],[6.,0.,0.],[10.,0.,0.]]),(3*n_beads,)), # four beads
      #x0 = jnp.reshape(jnp.array([[-2.,0.,0.]]),(3*n_beads,)), # SINGLE BEAD BENCHMARK
      #dimension = 3*n_beads,
      #noiseterms = 3*n_beads,
      tmax = 2000.0)

def moving_average(a, n):
    ret = jnp.cumsum(a)
    return (ret[n:] - ret[:-n])/n
    
def trace_mobility(x):
    locations = jnp.reshape(x,(n_beads,3))
    mu = pygrpy.jax_grpy_tensors.muTT(locations,radii).reshape(n_beads,3,n_beads,3)
    trace_mu = jnp.einsum('aibi -> ab', mu) # particle-wise trace
    return trace_mu
    
def optimal_weights(x):
    v_trace_mobility = jax.vmap(trace_mobility)
    trace_mu = jnp.mean(v_trace_mobility(x),axis=0)
    inv_trace_mu = jnp.linalg.inv(trace_mu)
    return jnp.sum(inv_trace_mu,axis=-1) / jnp.sum(inv_trace_mu)
    
solver = pychastic.sde_solver.SDESolver(dt = 0.05)
chunk_size = 1
trajectories = solver.solve_many(problem,n_trajectories = 2**10, chunk_size = chunk_size, chunks_per_randomization = 1)

weights = optimal_weights(trajectories["solution_values"][:,-1])

big_bead_displacement = trajectories["solution_values"][:,:,0:3] - trajectories["solution_values"][:,jnp.newaxis,0,0:3]
big_bead_distance = jnp.sum(big_bead_displacement**2,axis=-1)**0.5
big_bead_msd = jnp.mean(big_bead_distance**2,axis=0)
big_bead_instant_diffusion = jnp.ediff1d(big_bead_msd, to_end = float('nan')) / (solver.dt*chunk_size)

small_bead_displacement = trajectories["solution_values"][:,:,9:12] - trajectories["solution_values"][:,jnp.newaxis,0,9:12]
small_bead_distance = jnp.sum(small_bead_displacement**2,axis=-1)**0.5
small_bead_msd = jnp.mean(small_bead_distance**2,axis=0)
small_bead_instant_diffusion = jnp.ediff1d(small_bead_msd, to_end = float('nan')) / (solver.dt*chunk_size)

(s_len,t_len,d_len) = trajectories["solution_values"].shape
centre_trajectories = jnp.sum(trajectories["solution_values"].reshape(s_len,t_len,n_beads,3) * weights.reshape(1,1,n_beads,1),axis=2)
centre_displacement = centre_trajectories[:,:,:] - centre_trajectories[:,0,jnp.newaxis,:]
centre_distance = jnp.sum(centre_displacement**2,axis=-1)**0.5
centre_msd = jnp.mean(centre_distance**2,axis=0)
centre_instant_diffusion = jnp.ediff1d(centre_msd, to_end = float('nan')) / (solver.dt*chunk_size)


#
# plotting
#

window = 100

plt.plot(
    moving_average(trajectories["time_values"][0],n=window),
    moving_average(big_bead_instant_diffusion,n=window)
    )

plt.plot(
    moving_average(trajectories["time_values"][0],n=window),
    moving_average(small_bead_instant_diffusion,n=window)
    )
    
plt.plot(
    moving_average(trajectories["time_values"][0],n=window),
    moving_average(centre_instant_diffusion,n=window)
    )    

def beta_regression(a,b):
    return (jnp.corrcoef(jnp.array([a,b]))[0,1]) * jnp.std(b) / jnp.std(a)

big_bead_apparent = beta_regression(trajectories["time_values"][0],big_bead_msd)
small_bead_apparent = beta_regression(trajectories["time_values"][0],small_bead_msd)
centre_apparent = beta_regression(trajectories["time_values"][0],centre_msd)

print(f"{big_bead_apparent=}\n{small_bead_apparent=}\n{centre_apparent=}")

#plt.plot(trajectories[0]["time_values"],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,0:3]-sol[:,0,np.newaxis,0:3])**2,axis=2),axis=0),label='First, big bead') # big bead
#plt.plot(trajectories[0]["time_values"],(1.0/len(trajectories))*np.sum(np.sum((sol[:,:,9:12]-sol[:,0,np.newaxis,9:12])**2,axis=2),axis=0),label='Last, small bead') # small bead
# plt.plot([0.0,problem.tmax],[0.0,0.3333*(1.0/ma.pi)*problem.tmax]) # largest bead -- theory
#plt.plot([0.0,problem.tmax],[0.0,0.2898*(1.0/ma.pi)*problem.tmax]) # BD sim Cichocki et al -- theory
plt.plot([0.0,problem.tmax],[0.2898*(1.0/ma.pi),0.2898*(1.0/ma.pi)]) # BD sim Cichocki et al -- theory
plt.plot([0.0,problem.tmax],[0.2919*(1.0/ma.pi),0.2919*(1.0/ma.pi)]) # clever approximation Cichocki et al -- theory
plt.xlabel(r"Dimensionless time ($t/\tau$)")
plt.ylabel(r"Mean square displacement ($\mathbb{E}|q|^2$)")
#plt.legend()

plt.show()

