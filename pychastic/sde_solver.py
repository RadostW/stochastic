from functools import wraps
import jax
import jax.numpy as jnp
import numpy as np
from pychastic.sde_problem import SDEProblem

def get_wiener_integrals(key, steps=1, noise_terms=1):    
    return {
        "d_w": jnp.zeros((steps,1)),
        "d_www": jax.random.normal(key, shape=(steps, noise_terms))[:,:, jax.numpy.newaxis, jax.numpy.newaxis],
        
    }
        
def tensordot1(a, b):
    return jax.numpy.tensordot(a, b, axes=1)
        
def L_w_operator(f,problem):
    @wraps(f)
    def wrapped(x):
        return tensordot1(jax.jacobian(f)(x), problem.b(x))

    return wrapped


class SDESolver:

    def __init__(
        self,
        dt=0.01
    ):
        self.dt = dt

    def solve_many(self, problem: SDEProblem):
    
        n_trajectories = 2
        seed = 0
        dimension = 1
        noise_terms = 1
        steps_needed = 10
        chunk_size = 2
        number_of_chunks = 6
        chunks_per_randomization = 2
        
        t0 = 0.0
        w0 = jax.numpy.zeros(noise_terms)

        def L_w(f):
            return L_w_operator(f,problem)

        id_ = lambda x: x
        f_www = L_w(L_w(L_w(id_)))

        def step(
            x,
            d_t,
            d_w,
            d_www=jax.numpy.zeros((noise_terms, noise_terms, noise_terms)),
        ):

            new_x = (
                # x * d_www[0,0,0] # Ok
                # x * f_www(x)[0,0,0,0] # Ok
                jnp.array([1.]) * f_www(x)[0,0,0,0] * d_www[0,0,0] # AttributeError
                #x * f_www(x)[0,0,0,0] * d_www[0,0,0] # AttributeError
                # contract_all(f_www(x), d_www) # AttributeError
                # f_www(x)[:,0,0,0] # AttributeError
            )

            return new_x
        
        key = jax.random.PRNGKey(0)
        
        def scan_func(carry, input_):
            t, x, w = carry
            x = step(x, d_t=self.dt, d_w = input_['d_w'], d_www = input_['d_www'])
            return (t, x, w), (t, x, w)

        def chunk_function(chunk_start, wieners_chunk):
            # Parameters: chunk_start = (t0, x0, w0) values at beggining of chunk
            #             wieners_chunk = array of wiener increments
            z = jax.lax.scan( scan_func , chunk_start , wieners_chunk )[0] #discard trajectory at chunk resolution
            return z, z

        def get_solution_fragment(starting_state,key):
            wiener_integrals = get_wiener_integrals(key, steps=chunk_size*chunks_per_randomization, noise_terms=noise_terms)    

            last_state , (time_values, solution_values, wiener_values) = jax.lax.scan(
                chunk_function,
                starting_state,
                jax.tree_map(lambda x: jnp.reshape(x,(-1,chunk_size)+x.shape[1:]), wiener_integrals)
            ) #discard carry, remember trajectory

            return (last_state, last_state)

        @jax.vmap
        def get_solution(key):
            _ , chunked_solution = jax.lax.scan(
                lambda state, key: get_solution_fragment(state,key),
                (t0,problem.x0,w0),
                jax.random.split(key, 1)
                )

            return chunked_solution

        keys = jax.random.split(key, 1)
        solutions = get_solution(keys)

        return solutions
        
