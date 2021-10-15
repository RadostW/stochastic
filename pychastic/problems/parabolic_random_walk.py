# Regular 2D random walk, just in polar coordinates

from ..sde_problem import VectorSDEProblem
import jax
import numpy as np

ParabolicRandomWalk = VectorSDEProblem(
    lambda x: jnp.array([
        (x[0] - x[1]) / (2*(x[0]+x[1])**2),
        (-x[0] + x[1]) / (2*(x[0]+x[1])**2)
    ]),  
    lambda x: jnp.array([
        [
        1 / (2*x[0]+2*x[1]), x[1] / (x[0]+x[1])
        ],           
        [
        -1 / (2*x[0]+2*x[1]), x[0] / (x[0]+x[1])
        ] 
    ]),
    dimension = 2,
    noiseterms= 2,
    x0 = jnp.array([1.0,0.0]), # u=1 , v=0
    tmax=1.0
    )

# x = u^2 - v^2
# y = u + v
