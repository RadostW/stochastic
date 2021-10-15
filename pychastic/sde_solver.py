from copy import deepcopy
import jax
import time
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pychastic.sde_problem import SDEProblem
import pychastic.utils
import pychastic.wiener_integral_moments
from functools import wraps
import jax
from pychastic.vectorized_I_generation import get_wiener_integrals
import tqdm


def contract_all(a, b):
    return jax.numpy.tensordot(a, b, axes=len(b.shape))


def tensordot1(a, b):
    return jax.numpy.tensordot(a, b, axes=1)


def tensordot2(a, b):
    return jax.numpy.tensordot(a, b, axes=2)


class SDESolver:
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

    def solve_many(self, problem: SDEProblem, n_trajectories=1, seed=0):
        """
        Solves SDE problem given by ``problem``. Integration parameters are controlled by attribues of ``VectorSDESolver`` object.

        Parameters
        ----------
        problem : VectorSDEProblem
            SDE problem to be solved.
        seed : int, optional
            value of seed for PRNG.

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
            x,
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

            new_x = x
            new_x += (f_t(x)*d_t).squeeze() + contract_all(f_w(x), d_w)
            if scheme == "euler":
                return new_x

            new_x += contract_all(f_ww(x), d_ww)
            if scheme == "milstein":
                return new_x

            new_x += (
                contract_all(f_tw(x), d_tw)
                + contract_all(f_wt(x), d_wt)
                + contract_all(f_www(x), d_www)
            )
            if scheme == "wagner_platen":
                return new_x

        chunk_size = int(problem.tmax / self.dt)
        key = jax.random.PRNGKey(seed)
        
        def scan_func(carry, input_):
            t, x, w = carry
            
            wiener_integrals = input_
            wiener_integrals['d_w'] *= jax.numpy.sqrt(self.dt)
            wiener_integrals['d_ww'] *= self.dt
            
            t += self.dt
            x = step(x, d_t=self.dt, scheme=self.scheme, **wiener_integrals)
            w += wiener_integrals['d_w']
            return (t, x, w), (t, x, w)

        t0 = 0.0
        w0 = jax.numpy.zeros(noise_terms)

        @jax.vmap
        def get_solution(key):
            wiener_integrals = get_wiener_integrals(key, steps=chunk_size, noise_terms=noise_terms, scheme=self.scheme)
            _, (time_values, solution_values, wiener_values) = jax.lax.scan(scan_func, (t0, problem.x0, w0), wiener_integrals)
            
            return dict(
                time_values=time_values,
                solution_values=solution_values,
                wiener_values=wiener_values
            )

        keys = jax.random.split(key, n_trajectories)
        solutions = get_solution(keys)
        #return [{k: v[i] for k, v in solutions.items()} for i in range(n_trajectories)]  # TODO: Radost make this quick

        # Dis better. Cannot slice list(dict(list)), can slice dict(list(list)) in a sensible way.
        # For example:
        # >>> solution['solution_values'][:,-1]
        return solutions

    def solve(self, problem, seed=0):
        solution = self.solve_many(problem, n_trajectories=1, seed=seed)
        solution = jax.tree_map(lambda x: x[0], solution)
        return solution

if __name__ == '__main__':
    a = 1
    b = 1
    scalar_geometric_bm = SDEProblem(
        a = lambda x: a*x,
        b = lambda x: b*x,
        x0 = 1.0,
        tmax = 1.0,
        exact_solution = lambda x0, t, w: x0*np.exp((a-0.5*b*b)*t+b*w)   
    )
    problem = scalar_geometric_bm
    solver = SDESolver()
    steps = 100
    dt = problem.tmax / steps
    solver.dt = dt
    solver.solve_many(problem, n_trajectories=1000)
