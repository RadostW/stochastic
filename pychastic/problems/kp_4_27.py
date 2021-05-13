# Kloeden-Platen 4.27
# a=1.0
from ..sde_problem import SDEProblem
import jax
import numpy as np

a = 1.0
KloedenPlaten4_27 = SDEProblem(
    a=lambda x: -a**2*jax.numpy.sin(x)*jax.numpy.cos(x)**3,
    b=lambda x: a*jax.numpy.cos(x)**2,
    x0=0.0,
    tmax=1.0,
    exact_solution=lambda x0, t, w: np.arctan(a*w+np.tan(x0))
    )
