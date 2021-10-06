# Regular 2D random walk, just in polar coordinates

from ..sde_problem import VectorSDEProblem
import jax
import numpy as np

PolarRandomWalk = VectorSDEProblem(
    lambda x: jnp.array([1/(2*x[0]),0]),       # [1/2r,0]
    lambda x: jnp.array([
        [jnp.cos(x[1]),jnp.sin(x[1])],           # cos \phi,      sin \phi
        [-jnp.sin(x[1])/x[0],jnp.cos(x[1])/x[0]] # -sin \phi / r, cos \phi / r
    ]),
    dimension = 2,
    noiseterms= 2,
    x0 = jnp.array([1.0,0.0]), # r=1.0, \phi=0.0
    tmax=1.0
    )
