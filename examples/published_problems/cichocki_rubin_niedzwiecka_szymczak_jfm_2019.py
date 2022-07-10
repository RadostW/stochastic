# Following code is implementation of simulation published in:
# "Diffusion coefficients of elastic macromolecules"
# B. Cichocki, M. Rubin, A. Niedzwiecka & P. Szymczak
# J. Fluid Mech. (2019)
# doi:10.1017/jfm.2019.652

import pychastic  # solving sde
import pygrpy.jax_grpy_tensors  # hydrodynamic interactions
import jax.numpy as jnp  # jax array operations
import jax  # taking gradients
import matplotlib.pyplot as plt  # plotting
import numpy as np  # post processing trajectory
import math as ma  # math.pi
from tqdm import tqdm  # progess bar

radii = jnp.array([3.0, 1.0, 1.0, 1.0])  # sizes of spheres we're using
# radii = jnp.array([1.0]) # SINGLE BEAD BENCHMARK
n_beads = len(radii)
equilibrium_dist = 4.0
spring_constant = 5.5


def u_ene(x):  # potential energy shape
    # return 0.0 # SINGLE BEAD BENCHMARK
    locations = jnp.reshape(x, (n_beads, 3))
    distance_ab = jnp.sqrt(jnp.sum((locations[0] - locations[1]) ** 2))
    distance_bc = jnp.sqrt(jnp.sum((locations[1] - locations[2]) ** 2))
    distance_cd = jnp.sqrt(jnp.sum((locations[2] - locations[3]) ** 2))
    ene = (
        spring_constant * (distance_ab - equilibrium_dist) ** 2
        + spring_constant * (distance_bc - equilibrium_dist) ** 2
        + spring_constant * (distance_cd - equilibrium_dist) ** 2
    )
    return ene


def drift(x):
    locations = jnp.reshape(x, (n_beads, 3))
    mu = pygrpy.jax_grpy_tensors.muTT(locations, radii)
    force = -jax.grad(u_ene)(x)
    return jnp.matmul(mu, force)


def noise(x):
    locations = jnp.reshape(x, (n_beads, 3))
    mu = pygrpy.jax_grpy_tensors.muTT(locations, radii)
    return jnp.sqrt(2) * jnp.linalg.cholesky(mu)


problem = pychastic.sde_problem.SDEProblem(
    drift,
    noise,
    x0=jnp.reshape(
        jnp.array(
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        (3 * n_beads,),
    ),
    tmax=8000.0
    #tmax=800.0
)


# Compute trajectories

solver = pychastic.sde_solver.SDESolver(dt=0.2)
chunk_size = 40
trajectories = solver.solve_many(
    problem, n_trajectories=2 ** 10, chunk_size=chunk_size, chunks_per_randomization=1, seed = 2
)


# Post processing


def trace_mobility(x):
    locations = jnp.reshape(x, (n_beads, 3))
    mu = pygrpy.jax_grpy_tensors.muTT(locations, radii).reshape(n_beads, 3, n_beads, 3)
    trace_mu = jnp.einsum("aibi -> ab", mu)  # particle-wise trace
    return trace_mu


def optimal_weights(x):
    v_trace_mobility = jax.vmap(trace_mobility)
    trace_mu = jnp.mean(v_trace_mobility(x), axis=0)
    inv_trace_mu = jnp.linalg.inv(trace_mu)
    return jnp.sum(inv_trace_mu, axis=-1) / jnp.sum(inv_trace_mu)


def mstdc(x):
    # Minimal short time diffusion coefficient
    v_trace_mobility = jax.vmap(trace_mobility)
    trace_mu = jnp.mean(v_trace_mobility(x), axis=0)
    inv_trace_mu = jnp.linalg.inv(trace_mu)
    return 6.0 / jnp.sum(inv_trace_mu)


weights = optimal_weights(trajectories["solution_values"][:, -1])
mstdc_estimate = mstdc(trajectories["solution_values"][:, -1])

print(f"Optimal weights {weights}")
print(f"Minimal short time diffusion coefficient {mstdc_estimate}")

# From trajectories directly


def moving_average(a, n):
    ret = jnp.cumsum(a)
    return (ret[n:] - ret[:-n]) / n


def least_squares(x, y):
    Sx = jnp.mean(x)
    Sy = jnp.mean(y)
    Sxx = jnp.mean(x ** 2)
    Sxy = jnp.mean(x * y)
    Syy = jnp.mean(y ** 2)
    delta = Sxx - Sx ** 2

    a = (Sxy - Sx * Sy) / delta
    b = (Sxx * Sy - Sx * Sxy) / delta

    vy = Syy - a * Sxy - b * Sy

    va = vy / delta
    vb = va * Sxx

    return {"a": a, "sigma_a": jnp.sqrt(va), "b": b, "sigma_b": jnp.sqrt(vb)}


big_bead_displacement = (
    trajectories["solution_values"][:, :, 0:3]
    - trajectories["solution_values"][:, jnp.newaxis, 0, 0:3]
)
big_bead_distance = jnp.sum(big_bead_displacement ** 2, axis=-1) ** 0.5
big_bead_msd = jnp.mean(big_bead_distance ** 2, axis=0)
big_bead_instant_diffusion = jnp.ediff1d(big_bead_msd, to_end=float("nan")) / (
    solver.dt * chunk_size
)

small_bead_displacement = (
    trajectories["solution_values"][:, :, 9:12]
    - trajectories["solution_values"][:, jnp.newaxis, 0, 9:12]
)
small_bead_distance = jnp.sum(small_bead_displacement ** 2, axis=-1) ** 0.5
small_bead_msd = jnp.mean(small_bead_distance ** 2, axis=0)
small_bead_instant_diffusion = jnp.ediff1d(small_bead_msd, to_end=float("nan")) / (
    solver.dt * chunk_size
)

(s_len, t_len, d_len) = trajectories["solution_values"].shape
centre_trajectories = jnp.sum(
    trajectories["solution_values"].reshape(s_len, t_len, n_beads, 3)
    * weights.reshape(1, 1, n_beads, 1),
    axis=2,
)
centre_displacement = (
    centre_trajectories[:, :, :] - centre_trajectories[:, 0, jnp.newaxis, :]
)
centre_distance = jnp.sum(centre_displacement ** 2, axis=-1) ** 0.5
centre_msd = jnp.mean(centre_distance ** 2, axis=0)
centre_instant_diffusion = jnp.ediff1d(centre_msd, to_end=float("nan")) / (
    solver.dt * chunk_size
)

big_bead_apparent = least_squares(trajectories["time_values"][0], big_bead_msd)
small_bead_apparent = least_squares(trajectories["time_values"][0], small_bead_msd)
centre_apparent = least_squares(trajectories["time_values"][0], centre_msd)

print(f"{big_bead_apparent=}\n{small_bead_apparent=}\n{centre_apparent=}")

# From autocorelation function


@jax.jit
def average_diagonals(mat):
    n = mat.shape[0]

    # shift line n by n slots with ravel trick
    stairs = (
        (jnp.pad(jnp.flip(mat, axis=0), ((0, 0), (0, n))).ravel())[: n * (2 * n - 1)]
    ).reshape(n, 2 * n - 1)

    weights = jnp.hstack([jnp.arange(1, n), jnp.arange(n, 0, -1)])
    return jnp.sum(stairs, axis=0) / weights


def diffusion_estimate(sample):
    velocities = jnp.diff(sample) / (solver.dt * chunk_size)
    covariance = jnp.cov(velocities.T)
    acf = average_diagonals(covariance)  # autocovariance function
    return jnp.sum(acf) * (solver.dt * chunk_size)


centre_acf_estimate = (
    diffusion_estimate(centre_displacement[:, :, 0])
    + diffusion_estimate(centre_displacement[:, :, 1])
    + diffusion_estimate(centre_displacement[:, :, 2])
)

print(f"{centre_acf_estimate=} {centre_acf_estimate * ma.pi=}")


print("===========")
print(f"Minimal short time: {mstdc_estimate:.5f}")
print(f"OLS fit to rmsd:    {float(centre_apparent['a']) * ma.pi:.5f} +- {float(centre_apparent['sigma_a']) * ma.pi:.5f}")
print(f"ACF method:         {centre_acf_estimate * ma.pi:.5f}")
print(f"JFM reported:       {0.2898:.5f}")
print("===========")

#
# plotting
#

window = 1

plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(big_bead_instant_diffusion, n=window),
)

plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(small_bead_instant_diffusion, n=window),
)

plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(centre_instant_diffusion, n=window),
)

plt.plot(
    [0.0, problem.tmax], [0.2898 * (1.0 / ma.pi), 0.2898 * (1.0 / ma.pi)]
)  # BD sim Cichocki et al -- theory
plt.plot(
    [0.0, problem.tmax], [0.2919 * (1.0 / ma.pi), 0.2919 * (1.0 / ma.pi)]
)  # clever approximation Cichocki et al -- theory
plt.xlabel(r"Dimensionless time ($t/\tau$)")
plt.ylabel(r"Apparent diffusion coefficient")

plt.show()


plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(small_bead_msd, n=window),
)

plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(big_bead_msd, n=window),
)

plt.plot(
    moving_average(trajectories["time_values"][0], n=window),
    moving_average(centre_msd, n=window),
)
plt.show()
