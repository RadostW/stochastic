from functools import wraps
import jax
import jax.numpy as jnp
import numpy as np
from pychastic.sde_problem import SDEProblem

def get_wiener_integrals(key, steps=1, noise_terms=1):    
    return {
        "d_w": jnp.zeros((steps,1)),
        "d_www": jax.random.normal(key, shape=(steps, noise_terms,1,1)),
        
    }

def branch_fun(q):
    x = q[0]
    ret = jax.lax.cond(
    x > 1.0,
    lambda y: jnp.exp(x)   ,
    lambda y: jnp.exp(x)
    ,x)
    return jnp.array([[ret]])

        
def tensordot1(a, b):
    return jax.numpy.tensordot(a, b, axes=1)
        
def L_w_operator(f):
    @wraps(f)
    def wrapped(x):
        return tensordot1(jax.jacobian(f)(x), branch_fun(x))

    return wrapped


class SDESolver:

    def __init__(
        self,
        dt=0.01
    ):
        self.dt = dt

    def solve_many(self):
        noise_terms = 1
                
        t0 = 0.0
        w0 = jax.numpy.zeros(noise_terms)

        def L_w(f):
            return L_w_operator(f)

        id_ = lambda x: x
        f_www = L_w(L_w(L_w(id_)))

        def step(
            x,
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
            x = step(x, d_www = input_['d_www'])
            return (t, x, w), (t, x, w)

        def chunk_function(chunk_start, wieners_chunk):
            z = jax.lax.scan( scan_func , chunk_start , wieners_chunk )[0]
            return z, z

        def get_solution_fragment(starting_state,key):
            wiener_integrals = get_wiener_integrals(key, steps=2*2, noise_terms=noise_terms)    

            last_state , (time_values, solution_values, wiener_values) = jax.lax.scan(
                chunk_function,
                starting_state,
                jax.tree_map(lambda x: jnp.reshape(x,(-1,2)+x.shape[1:]), wiener_integrals)
            ) #discard carry, remember trajectory

            return (last_state, last_state)

        @jax.vmap
        def get_solution(key):
            _ , chunked_solution = jax.lax.scan(
                lambda state, key: get_solution_fragment(state,key),
                (t0,jnp.array([0.1]),w0),
                jax.random.split(key, 1)
                )

            return chunked_solution

        keys = jax.random.split(key, 1)
        solutions = get_solution(keys)

        return solutions
        
