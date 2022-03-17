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
        drift, noise, tmax=5., x0=jnp.array([2.0])
    )

    solver = pychastic.sde_solver.SDESolver(dt=0.5)

    trajectories = solver.solve_many()
