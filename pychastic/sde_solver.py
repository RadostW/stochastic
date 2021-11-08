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
from jax.experimental.host_callback import id_tap

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

    def solve_many(self, problem: SDEProblem, n_trajectories=1, seed=0, chunk_size=1, chunks_per_randomization = None, progress_bar = True):
        """
        Solves SDE problem given by ``problem``. Integration parameters are controlled by attribues of ``SDESolver`` object.

        Parameters
        ----------
        problem : SDEProblem
            (Vector) SDE problem to be solved.
        seed : int, optional
            value of seed for PRNG.
        chunk_size: int or None, optional
            Make steps in solver in chunks of `chunk_size` steps.
            If `chunk_size = n` then value at every nth step is returned.
            If `None` then maximal size of chunk is used only final value is returned.
        chunks_per_randomization: int or None, optional
            Sample wiener trajectories once per `chunks_per_randomization` chunks.
            If `chunks_per_randomization = n` then PRNG runs at every nth chunk.
            If `None`, PRNG runs once, at beginning of simulation.
            Smaller values lead to less memory usage. Larger values increase speed.
        progress_bar: True, False
            Display `tqdm` style progress bar during computation.

        Returns
        -------
        dict
            Under following keys in returned `dict` you'll find:

            * ``time_values`` -- (`n_trajectories`, `steps`) jnp.array containing timestamps coresponding to each trajectory.
            * ``solution_values`` -- (`n_trajectories`, `steps`, `problem_dimension`) `jnp.array` containing values of integrated SDE.
            * ``wiener_values`` -- (`n_trajectories`, `steps`, `noise_dimension`) `jnp.array` containing values of Wiener processes driving the SDE.

        Example
        -------
        >>> import pychastic
        >>> import jax.numpy as jnp
        >>> solver = pychastic.sde_solver.SDESolver()
        >>> problem = pychastic.sde_problem.SDEProblem(
        ... lambda x: jnp.array([1/(2*x[0]),0]),       # [1/2r,0]
        ... lambda x: jnp.array([
        ...    [jnp.cos(x[1]),jnp.sin(x[1])],           # cos \phi,      sin \phi
        ...    [-jnp.sin(x[1])/x[0],jnp.cos(x[1])/x[0]] # -sin \phi / r, cos \phi / r
        ... ]),
        ... x0 = jnp.array([1.0,0.0]), # r=1.0, \phi=0.0
        ... tmax=0.02
        ... )
        >>> solution = solver.solve_many(problem,n_trajectories = 100)
        >>> (r,phi) = (solution["solution_values"][:,-1,:]).transpose()
        >>> compare = {"integrated":(r*jnp.array([jnp.cos(phi),jnp.sin(phi)])).T,"exact":solution["wiener_values"][:,-1,:]+problem.x0}
        >>> print(compare["integrated"][0],compare["exact"][0])
            
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

        steps_needed = int(problem.tmax / self.dt)
        tmp = chunks_per_randomization or 1
        chunk_size = chunk_size or steps_needed
        number_of_chunks = ((steps_needed // (chunk_size*tmp)) + (1 if steps_needed % (chunk_size*tmp) else 0))*tmp
        chunks_per_randomization = chunks_per_randomization or number_of_chunks

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

        if progress_bar:
            p_bar = tqdm.tqdm(total = number_of_chunks)
            def tap_func(*args,**kwargs):
                p_bar.update()
        else:
            def tap_func(*args,**kwargs):
                pass

        def chunk_function(chunk_start, wieners_chunk):
            # Parameters: chunk_start = (t0, x0, w0) values at beggining of chunk
            #             wieners_chunk = array of wiener increments
            id_tap(tap_func,0)
            z = jax.lax.scan( scan_func , chunk_start , wieners_chunk )[0] #discard trajectory at chunk resolution
            return z, z

        def get_solution_fragment(starting_state,key):
            wiener_integrals = get_wiener_integrals(key, steps=chunk_size*chunks_per_randomization, noise_terms=noise_terms, scheme=self.scheme)    

            last_state , (time_values, solution_values, wiener_values) = jax.lax.scan( 
                chunk_function, 
                starting_state,
                jax.tree_map(lambda x: jnp.reshape(x,(-1,chunk_size)+x.shape[1:]), wiener_integrals)
            ) #discard carry, remember trajectory
            
            return (
                    last_state,
                    dict(
                    time_values=time_values,
                    solution_values=solution_values,
                    wiener_values=wiener_values,
                    )
                   )

        @jax.vmap
        def get_solution(key):
            _ , chunked_solution = jax.lax.scan(
                lambda state, key: get_solution_fragment(state,key),
                (t0,problem.x0,w0),
                jax.random.split(key, number_of_chunks // chunks_per_randomization)
                )

            return jax.tree_map(lambda x: x.reshape((-1,)+x.shape[2:]),chunked_solution) #combine big chunks into one trajectory

        keys = jax.random.split(key, n_trajectories)
        solutions = get_solution(keys)
        if progress_bar: 
            p_bar.refresh()
        return solutions

    def solve(self, problem, seed=0, chunk_size=1, chunks_per_randomization = None, progress_bar = True):
        """
        Solves SDE problem given by ``problem``. Integration parameters are controlled by attribues of ``SDESolver`` object.

        Parameters
        ----------
        problem : SDEProblem
            (Vector) SDE problem to be solved.
        seed : int, optional
            value of seed for PRNG.
        chunk_size: int or None, optional
            Make steps in solver in chunks of `chunk_size` steps.
            If `chunk_size = n` then value at every nth step is returned.
            If `None` then maximal size of chunk is used only final value is returned.
        chunks_per_randomization: int or None, optional
            Sample wiener trajectories once per `chunks_per_randomization` chunks.
            If `chunks_per_randomization = n` then PRNG runs at every nth chunk.
            If `None`, PRNG runs once, at beginning of simulation.
            Smaller values lead to less memory usage. Larger values increase speed.
        progress_bar: True, False
            Display `tqdm` style progress bar during computation.

        Returns
        -------
        dict
            Under following keys in returned `dict` you'll find:

            * ``time_values`` -- (`steps`,) jnp.array containing timestamps coresponding to each trajectory.
            * ``solution_values`` -- (`steps`, `problem_dimension`) `jnp.array` containing values of integrated SDE.
            * ``wiener_values`` -- (`steps`, `noise_dimension`) `jnp.array` containing values of Wiener processes driving the SDE.

        Example
        -------
        >>> import pychastic
        >>> import jax.numpy as jnp
        >>> solver = pychastic.sde_solver.SDESolver()
        >>> problem = pychastic.sde_problem.SDEProblem(
        ... lambda x: jnp.array([1/(2*x[0]),0]),       # [1/2r,0]
        ... lambda x: jnp.array([
        ...    [jnp.cos(x[1]),jnp.sin(x[1])],           # cos \phi,      sin \phi
        ...    [-jnp.sin(x[1])/x[0],jnp.cos(x[1])/x[0]] # -sin \phi / r, cos \phi / r
        ... ]),
        ... x0 = jnp.array([1.0,0.0]), # r=1.0, \phi=0.0
        ... tmax=0.02
        ... )
        >>> solution = solver.solve(problem)
        >>> (r,phi) = (solution["solution_values"][-1,:])
        >>> compare = {"integrated":(r*jnp.array([jnp.cos(phi),jnp.sin(phi)])).T,"exact":solution["wiener_values"][-1,:]+problem.x0}
        >>> print(compare["integrated"],compare["exact"])
            
        """
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
