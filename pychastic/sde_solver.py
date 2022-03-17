from functools import wraps
import jax
import jax.numpy as jnp
import numpy as np
from pychastic.sde_problem import SDEProblem

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
                
        t0 = 0.0
        w0 = jax.numpy.zeros(1)

        def L_w(f):
            return L_w_operator(f)

        id_ = lambda x: x
        f_www = L_w(L_w(L_w(id_)))

        def step(
            x,
            d_www,
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
        
        def scan_func(x, y):
            xp = step(x, y['d_www'])
            return (xp,xp)

        def chunk_function(chunk_start, wieners_chunk):
            z = jax.lax.scan( scan_func , chunk_start , wieners_chunk )[0]
            return z, z

        def get_solution_fragment(starting_state,key):
            wiener_integrals = { "d_www": jax.random.normal(key, shape=(4,1,1,1)) }

            last_state , solution_values = jax.lax.scan(
                chunk_function,
                starting_state,
                jax.tree_map(lambda x: jnp.reshape(x,(-1,2)+x.shape[1:]), wiener_integrals)
            ) #discard carry, remember trajectory

            return (last_state, last_state)

        @jax.vmap
        def get_solution(key):
            _ , chunked_solution = jax.lax.scan(
                lambda state, key: get_solution_fragment(state,key),
                (jnp.array([0.1])),
                jax.random.split(key, 1)
                )

            return chunked_solution

        keys = jax.random.split(key, 1)
        solutions = get_solution(keys)

        return solutions
        
