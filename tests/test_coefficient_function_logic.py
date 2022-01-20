import jax
import jax.numpy as jnp
import pychastic.sde_problem
from functools import wraps

# Check logic inside sde_solver
# Code is copied because L_t L_w and L are local variables and are hard to test
# Such design is chosed because they depend on problem.a and problem.b

problem = pychastic.sde_problem.SDEProblem(
a = lambda x : jnp.exp(3.*x),
b = lambda x : jnp.exp(7.*x),
tmax = 1.0,
x0 = jnp.array(0.)
)

from pychastic.sde_solver import tensordot1
from pychastic.sde_solver import tensordot2

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
    for x in reversed(idx):
        if x == "t":
            f = L_t(f)
        elif x == "w":
            f = L_w(f)
        else:
            raise ValueError
    return f
    
def test_L_operator_logic():
    f = lambda x : x
    x0 = jnp.array([0.0])

    f_t = L(f,"t")(x0).squeeze() # should be a
    f_w = L(f,"w")(x0).squeeze() # should be b
    f_ww = L(f,"ww")(x0).squeeze() # should be b*b'
    f_tw = L(f,"tw")(x0).squeeze() # should be a*b' + 0.5*b*b''
    f_wt = L(f,"wt")(x0).squeeze() # should be b*a'
    
    sample_coefficients = jnp.array([f_t,f_w,f_ww,f_tw,f_wt])
    target_coefficients = jnp.array([1.,1.,7.,7.+0.5*7.*7.,3.])
    
    assert jnp.allclose(sample_coefficients,target_coefficients)
        
if __name__ == "__main__":
    test_L_operator_logic()        
