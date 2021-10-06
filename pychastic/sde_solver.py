import jax
import time
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pychastic.sde_problem import SDEProblem
from pychastic.sde_problem import VectorSDEProblem
from pychastic.wiener import VectorWienerWithI, Wiener
from pychastic.wiener import WienerWithZ
from pychastic.wiener import VectorWiener
import pychastic.utils
import pychastic.wiener_integral_moments
from functools import wraps
import jax
from pychastic.vectorized_I_generation import get_wiener_integrals


def contract_all(a, b):
    return jax.numpy.tensordot(a, b, axes=len(b.shape))


def tensordot1(a, b):
    return jax.numpy.tensordot(a, b, axes=1)


def tensordot2(a, b):
    return jax.numpy.tensordot(a, b, axes=2)


class VectorSDESolver:
    """
    Produces realisations of stochastic process to ``solve`` method.
    Controls numerical integration features via attributes.

    Parameters
    ----------
    scheme : {'euler', 'commutative_milstein', 'milstein'}, default: 'euler'
         Type of scheme used for integration.
    dt : float
         Step size in fixed-step integration.

    """

    def __init__(
        self,
        scheme="euler",
        dt=0.01,
        dt_adapting_factor=10,
        min_dt=None,
        max_dt=None,
        error_terms=1,
        target_mse_density=1e-2,
        adaptive=False,
    ):
        self.scheme = scheme  # euler | milstein
        self.dt = dt
        self.min_dt = min_dt or self.dt / dt_adapting_factor
        self.max_dt = max_dt or self.dt * dt_adapting_factor
        self.error_terms = error_terms
        self.target_mse_density = target_mse_density
        self.adaptive = adaptive

    def solve(self, problem: VectorSDEProblem):
        """
        Solves SDE problem given by ``problem``. Integration parameters are controlled by attribues of ``VectorSDESolver`` object.

        Parameters
        ----------
        problem : VectorSDEProblem
            SDE problem to be solved.

        Returns
        -------
        dict
            Under following keys you'll find:
            ``last_time`` -- time at last step of integration
            ``last_value`` -- value at last step of integration
            ``last_wiener`` -- value of underlying wiener process at last step of integration
            ``trajectory`` -- a jnp.array containing entire trajectory of the process, each entry in the array consists of 3 elements (time, value, wieners).

        Example
        -------
        >>> solver = VectorSDESolver()
        >>> problem = pychastic.sde_problem.VectorSDEProblem(
        ... lambda x: jnp.array([1/(2*x[0]),0]),       # [1/2r,0]
        ... lambda x: jnp.array([
        ...    [jnp.cos(x[1]),jnp.sin(x[1])],           # cos \phi,      sin \phi
        ...    [-jnp.sin(x[1])/x[0],jnp.cos(x[1])/x[0]] # -sin \phi / r, cos \phi / r
        ... ]),
        ... dimension = 2,
        ... noiseterms= 2,
        ... x0 = jnp.array([1.0,0.0]), # r=1.0, \phi=0.0
        ... tmax=0.02
        ... )
        >>> solution = solver.solve(problem)
        >>> (r,phi) = solution["last_value"]
        >>> compare = {"integrated":r*jnp.array([jnp.cos(phi),jnp.sin(phi)]),"exact":solution["last_wiener"]+problem.x0}
        >>> print(compare)
            
        """

        assert problem.x0.shape == problem.a(problem.x0).shape
        assert problem.x0.shape[0] == problem.b(problem.x0).shape[0]

        dimension, noise_terms = problem.b(problem.x0).shape

        def L_t(f):
            @wraps(f)
            def wrapped(x):
                b_val = problem.b(x)
                val = tensordot1(jax.jacobian(f)(x), problem.a(x)) + 0.5 * tensordot2(
                    jax.hessian(f)(x), tensordot1(b_val, b_val.T)
                )
                val = jax.numpy.expand_dims(val, -1)
                return val

            return wrapped

        def L_w(f):
            @wraps(f)
            def wrapped(x):
                return tensordot1(jax.jacobian(f)(x), problem.b(x))

            return wrapped

        def L(f, idx):
            for x in idx:
                if x == "t":
                    f = L_t(f)
                elif x == "w":
                    f = L_w(f)
                else:
                    raise ValueError
            return f

        def step(
            x_before,
            d_t,
            d_w,
            d_ww=jax.numpy.zeros((noise_terms, noise_terms)),
            d_tw=jax.numpy.zeros((1, noise_terms)),
            d_wt=jax.numpy.zeros((noise_terms, 1)),
            d_www=jax.numpy.zeros((noise_terms, noise_terms, noise_terms)),
            scheme="euler",
        ):

            id_ = lambda x: x
            f_t = L(id_, "t")
            f_w = L(id_, "w")
            f_ww = L(id_, "ww")
            f_tw = L(id_, "tw")
            f_wt = L(id_, "wt")
            f_www = L(id_, "www")

            x_after = x_before

            x_after += (f_t(x_before)*d_t).squeeze() + contract_all(f_w(x_before), d_w)
            if scheme == "euler":
                return x_after

            x_after += contract_all(f_ww(x_before), d_ww)
            if scheme == "milstein":
                return x_after

            x_after += (
                contract_all(f_tw(x_before), d_tw)
                + contract_all(f_wt(x_before), d_wt)
                + contract_all(f_www(x_before), d_www)
            )
            if scheme == "wagner_platen":
                return x_after

        chunk_size = int(problem.tmax / self.dt) + 1
        seed = 0
        key = jax.random.PRNGKey(seed)
        wiener_integrals = get_wiener_integrals(key, steps=chunk_size, noise_terms=noise_terms, scheme='euler')

        def scan_func(carry, input_):
            t, x, w = carry
            
            wiener_integrals = input_
            wiener_integrals['d_w'] *= jax.numpy.sqrt(self.dt)
            wiener_integrals['d_ww'] *= self.dt
            
            t += self.dt
            x = step(x, d_t=self.dt, **wiener_integrals)
            w += wiener_integrals['d_w']
            return (t, x, w), (t, x, w)

        t0 = 0.0
        w0 = jax.numpy.zeros(noise_terms)
        ((last_time,last_value,last_wiener),trajectory) = jax.lax.scan(jax.jit(scan_func), (t0, problem.x0, w0), wiener_integrals)
        return {"last_time":last_time,"last_value":last_value,"last_wiener":last_wiener,"trajectory":trajectory}


if __name__ == "__main__":
    #from jax.config import config
    #config.update('jax_disable_jit', True)

    solver = VectorSDESolver(dt=2**(-19))
    problem = pychastic.sde_problem.VectorSDEProblem(
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

    solution = solver.solve(problem)
    (r,phi) = solution["last_value"]
    compare = {"integrated":r*jnp.array([jnp.cos(phi),jnp.sin(phi)]),"exact":solution["last_wiener"]+problem.x0}
    print(compare)