# Following code is implementation of simulation published in:
# "Brownian Dynamics Simulations of Rotational Diffusion Using
# the Cartesian Components of the Rotation Vector as Generalized Coordinates"
# T. R. Evensen, S. N. Naess & A. Elgsaeter
# Macromol. Theory Simul. (2008)
# doi:10.1002/mats.200800031

import pychastic
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

mobility = 2.0*jnp.eye(3)
mobility_d = jnp.linalg.cholesky(mobility) # Compare with equation: Evensen2008.6

def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(q):
    # Compare with equation: Evensen2008.11
    phi = jnp.sqrt(jnp.sum(q ** 2))
    rot = (
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1)
    )
    return jax.lax.cond(phi > 0.01, lambda: rot, lambda: 1.0 * jnp.eye(3))


def transformation_matrix(q):
    # Compare with equation: Evensen2008.12
    phi = jnp.sqrt(jnp.sum(q ** 2))
    trans = (
        0.5
        * (1.0 / phi ** 2 - (jnp.sin(phi) / (2.0 * phi * (1.0 - jnp.cos(phi)))))
        * q.reshape(1, 3)
        * q.reshape(3, 1)
        + spin_matrix(q)
        + (phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))) * jnp.eye(3)
    )
    return jax.lax.cond(phi > 0.01, lambda: trans, lambda: 1.0 * jnp.eye(3))


def metric_force(q):
    # Compare with equation: Evensen2008.10
    phi = jnp.sqrt(jnp.sum(q ** 2))
    scale = jax.lax.cond(
        phi < 0.01,
        lambda t: -t / 6.0,
        lambda t: jnp.sin(t) / (1.0 - jnp.cos(t)) - 2.0 / t,
        phi,
    )
    return jax.lax.cond(
        phi > 0.0, lambda: (q / phi) * scale, lambda: jnp.array([0.0, 0.0, 0.0])
    )


def t_mobility(q):
    # Mobility matrix transformed to coordinates.
    # Compare with equation: Evensen2008.2
    return transformation_matrix(q) @ mobility @ (transformation_matrix(q).T)


def drift(q):
    # Drift term.
    # Compare with equation: Evensen2008.5 
    # jax.jacobian has differentiation index last (like mu_ij d_k) so divergence is contraction of first and last axis.
    return t_mobility(q) @ metric_force(q) + jnp.einsum(
        "iji->j", jax.jacobian(t_mobility)(q)
    )


def noise(q):
    # Noise term.
    # Compare with equation: Evensen2008.5
    return jnp.sqrt(2) * transformation_matrix(q) @ (rotation_matrix(q).T) @ mobility_d


def canonicalize_coordinates(q):
    phi = jnp.sqrt(jnp.sum(q ** 2))
    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi
    return jax.lax.cond(
        phi > max_phi,
        lambda canonical_phi, phi, q: (canonical_phi / phi) * q,
        lambda canonical_phi, phi, q: q,
        canonical_phi,
        phi,
        q,
    )


problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=20.0, x0=jnp.array([1.0, 0.0, 0.0])
)

solver = pychastic.sde_solver.SDESolver(dt=0.01)

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=1000,
    chunk_size=100,
    chunks_per_randomization=1,
)

final_angles = np.array(
    jnp.sqrt(jnp.sum(trajectories["solution_values"][:, -1, :] ** 2, axis=1))
)

final_x = np.array(trajectories["solution_values"][:, -1, 0])
final_y = np.array(trajectories["solution_values"][:, -1, 1])
final_z = np.array(trajectories["solution_values"][:, -1, 2])

xvals = np.arange(0, np.pi, 0.01)
yvals = (np.pi / 20) * len(final_angles) * ((1.0 - np.cos(xvals)) / np.pi)
plt.plot(xvals, yvals)
plt.hist(final_angles, 20)
# plt.hist(final_x, 20)
# plt.hist(final_y, 20)
# plt.hist(final_z, 20)
plt.show()
