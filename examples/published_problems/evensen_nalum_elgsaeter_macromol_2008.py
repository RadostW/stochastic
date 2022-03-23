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

mobility = jnp.eye(3)
mobility_d = jnp.linalg.cholesky(mobility)  # Compare with equation: Evensen2008.6


def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(q):
    # Compare with equation: Evensen2008.11
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    rot = (
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1)
    )

    return jnp.where(
        phi_squared == unsafe_phi_squared,
        rot,
        (1.0 - 0.5 * unsafe_phi_squared) * jnp.eye(3)
        + spin_matrix(q)
        + 0.5 * q.reshape(1, 3) * q.reshape(3, 1),
    )


def transformation_matrix(q):
    # Compare with equation: Evensen2008.12 - there are typos!
    # Compare with equation: Ilie2014.A9-A10 - no typos there
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    c = phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))

    trans = jnp.where(
        phi_squared == unsafe_phi_squared,
        ((1.0 - 0.5 * c) / (phi ** 2)) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + 0.5 * c * jnp.eye(3),
        (1.0 / 12.0) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + jnp.eye(3),
    )

    return trans


def metric_force(q):
    # Compare with equation: Evensen2008.10
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    scale = jnp.where(
        phi == unsafephi,
        jnp.sin(phi) / (1.0 - jnp.cos(phi)) - 2.0 / phi,
        -unsafephi / 6.0,
    )

    return jnp.where(phi == unsafephi, (q / phi) * scale, jnp.array([0.0, 0.0, 0.0]))


def t_mobility(q):
    # Mobility matrix transformed to coordinates.
    # Compare with equation: Evensen2008.2
    return (
        transformation_matrix(q)
        @ (rotation_matrix(q).T)
        @ mobility
        @ rotation_matrix(q)
        @ (transformation_matrix(q).T)
    )


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
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi

    return jax.lax.select(
        phi > max_phi,  # and phi == unsafephi
        (canonical_phi / phi) * q,
        q,
    )


problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=2.0, x0=jnp.array([0.0, 0.0, 0.0])
)


solver = pychastic.sde_solver.SDESolver(dt=0.1, scheme="euler")

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=10000,
    chunk_size=1,
    chunks_per_randomization=2,
)


rotation_matrices = jax.vmap(jax.vmap(rotation_matrix))(trajectories["solution_values"])
rotation_matrices = jnp.einsum(
    "ij,abjk", (rotation_matrix(problem.x0).T), rotation_matrices
)

epsilon_tensor = jnp.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
    ]
)

delta_u = -0.5 * jnp.einsum("kij,abij->abk", epsilon_tensor, rotation_matrices)

cor = jnp.mean(delta_u ** 2, axis=0)

t_a = trajectories["time_values"][0]
t_t = jnp.arange(0.0, trajectories["time_values"][0][-1], 0.005)
plt.plot(t_a, cor[:, 0])
plt.plot(t_a, cor[:, 1])
plt.plot(t_a, cor[:, 2])

D = 1.0
plt.plot(
    t_t,
    1.0 / 6.0
    - (5.0 / 12.0) * jnp.exp(-6.0 * D * t_t)
    + (1.0 / 4.0) * jnp.exp(-2.0 * D * t_t),
    label="theoretical",
)

plt.show()
