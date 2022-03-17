import pychastic
import jax.numpy as jnp
import jax

def drift(q):
    return jnp.array([1.0])

def noise(q):
    x = q[0]
    ret = jax.lax.cond(
    x > 1.0,
    lambda y: jnp.exp(x)   ,
    lambda y: jnp.exp(x)
    ,x)
    return jnp.array([[ret]])

def test_branched_coefficients():
    problem = pychastic.sde_problem.SDEProblem(
        drift, noise, tmax=1., x0=jnp.array([0.0001])
    )

    solver = pychastic.sde_solver.SDESolver(dt=0.5,scheme='wagner_platen')

    trajectories = solver.solve_many(
        problem,
        n_trajectories=2,
        chunk_size=2,
        chunks_per_randomization=2,
    )
