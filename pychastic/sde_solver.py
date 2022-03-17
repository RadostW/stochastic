from functools import wraps
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.experimental.host_callback import id_tap
from pychastic.sde_problem import SDEProblem
from pychastic.vectorized_I_generation import get_wiener_integrals
        
def tensordot1(a, b):
    return jax.numpy.tensordot(a, b, axes=1)

def tensordot2(a, b):
    return jax.numpy.tensordot(a, b, axes=2)
        
# Taylor-Ito expansion operators    
def L_t_operator(f,problem):
    @wraps(f)
    def wrapped(x):
        b_val = problem.b(x)
        val = tensordot1(jax.jacobian(f)(x), problem.a(x)) + 0.5 * tensordot2(
            jax.hessian(f)(x), tensordot1(b_val, b_val.T)
        )
        return val[:,jnp.newaxis,...] #indexing convention [spatial, time, ... = noiseterms/time]

    return wrapped

def L_w_operator(f,problem):
    @wraps(f)
    def wrapped(x):
        val =  tensordot1(jax.jacobian(f)(x), problem.b(x))[:,jnp.newaxis,...]
        return jnp.swapaxes(val,1,-1)[:,...,0] # indexing convention [spatial, noiseterms, ... = noiseterms/time]

    return wrapped


class SDESolver:
    """
    Produces realisations of stochastic process to ``solve`` method.
    Controls numerical integration features via attributes.

    Parameters
    ----------
    scheme : {'euler', 'milstein', 'wagner_platen'}, default: 'euler'
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

    def solve_many(self, problem: SDEProblem, n_trajectories=1, step_post_processing = None, seed=0, chunk_size=1, chunks_per_randomization = None, progress_bar = True):
        """
        Solves SDE problem given by ``problem``. Integration parameters are controlled by attribues of ``SDESolver`` object.

        Parameters
        ----------
        problem : SDEProblem
            (Vector) SDE problem to be solved.
        n_trajectories : int, optional
            Number of sample paths to generate
        step_post_processing : callable
            (Advanced) Function of with call signature f(x) returning canonical coordinates of x.
            Usefull when simulating process on a manifold with does not have covering map from :math:`\\mathbb{R}^n` such as :math:`SO(3)`.
            Post processing function has to `jit` with jax. To deal with branch cuts and such refer to `jax.lax.cond`.
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
        ...    [jnp.cos(x[1]),jnp.sin(x[1])],           # cos phi,      sin phi
        ...    [-jnp.sin(x[1])/x[0],jnp.cos(x[1])/x[0]] # -sin phi / r, cos phi / r
        ... ]),
        ... x0 = jnp.array([1.0,0.0]), # r=1.0, phi=0.0
        ... tmax=0.02
        ... )
        >>> solution = solver.solve_many(problem,n_trajectories = 100)
        >>> solution_principal = solver.solve_many(problem,step_post_processing = lambda x : jnp.fmod(x,2*jnp.pi),n_trajectories = 100)
        >>> (r,phi) = (solution["solution_values"][:,-1,:]).transpose()
        >>> compare = {"integrated":(r*jnp.array([jnp.cos(phi),jnp.sin(phi)])).T,"exact":solution["wiener_values"][:,-1,:]+problem.x0}
        >>> print(compare["integrated"][0],compare["exact"][0])
            
        """

        assert problem.x0.shape == problem.a(problem.x0).shape
        assert problem.x0.shape[0] == problem.b(problem.x0).shape[0]

        dimension, noise_terms = problem.b(problem.x0).shape

        def L_t(f):
            return L_t_operator(f,problem)

        def L_w(f):
            return L_w_operator(f,problem)

        def L(f, idx):
            for x in reversed(idx):
                if x == "t":
                    f = L_t(f)
                elif x == "w":
                    f = L_w(f)
                else:
                    raise ValueError
            return f


        id_ = lambda x: x

        f_www = L(id_, "www")

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

            new_x = x

            new_x += (
                # x * d_www[0,0,0] # Ok
                # x * f_www(x)[0,0,0,0] # Ok
                jnp.array([1.]) * f_www(x)[0,0,0,0] * d_www[0,0,0] # AttributeError
                #x * f_www(x)[0,0,0,0] * d_www[0,0,0] # AttributeError
                # contract_all(f_www(x), d_www) # AttributeError
                # f_www(x)[:,0,0,0] # AttributeError
            )

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
            wiener_integrals_rescaled = dict()
            wiener_integrals_rescaled['d_w'] = jax.numpy.sqrt(self.dt) * wiener_integrals['d_w']

            wiener_integrals_rescaled['d_www'] = 0*self.dt**(3/2) * wiener_integrals['d_www']
            #wiener_integrals_rescaled['d_www'] = jax.numpy.zeros((noise_terms, noise_terms, noise_terms))

            x = step(x, d_t=self.dt, scheme=self.scheme, d_w = wiener_integrals_rescaled['d_w'], d_www = wiener_integrals_rescaled['d_www'])

            return (t, x, w), (t, x, w)

        t0 = 0.0
        w0 = jax.numpy.zeros(noise_terms)

        def chunk_function(chunk_start, wieners_chunk):
            # Parameters: chunk_start = (t0, x0, w0) values at beggining of chunk
            #             wieners_chunk = array of wiener increments
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

            return chunked_solution
            #return jax.tree_map(lambda x: x.reshape((-1,)+x.shape[2:]),chunked_solution) #combine big chunks into one trajectory

        keys = jax.random.split(key, n_trajectories)
        solutions = get_solution(keys)

        return solutions
        
