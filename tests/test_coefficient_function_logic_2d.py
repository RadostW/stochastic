import jax.numpy as jnp
import pychastic.sde_problem

# Check logic inside sde_solver
# Code is copied because L_t L_w and L are local variables and are hard to test
# Such design is chosen because they depend on problem.a and problem.b

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: jnp.array([1.1 * jnp.exp(3.0 * x[0]), 1.2 * jnp.exp(5.0 * x[1])]),
    b=lambda x: jnp.array(
        [
            [1.3 * jnp.exp(7.0 * x[0]), 1.4 * jnp.exp(11.0 * x[1])],
            [1.5 * jnp.exp(13.0 * x[1]), 1.6 * jnp.exp(17.0 * x[0])],
        ]
    ),
    tmax=1.0,
    x0=jnp.array([0.0, 0.0]),
)

from pychastic.sde_solver import L_t_operator
from pychastic.sde_solver import L_w_operator

assert problem.x0.shape == problem.a(problem.x0).shape
assert problem.x0.shape[0] == problem.b(problem.x0).shape[0]

dimension, noise_terms = problem.b(problem.x0).shape


def L_t(f):
    return L_t_operator(f, problem)


def L_w(f):
    return L_w_operator(f, problem)


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
    f = lambda x: x
    x0 = problem.x0

    f_t = L(f, "t")(x0).squeeze()  # should be a
    f_w = L(f, "w")(x0).squeeze()  # should be b
    f_ww = L(f, "ww")(x0).squeeze()  # should be b*b'
    # f_tw = L(f, "tw")(x0).squeeze()  # should be a*b' + 0.5*b*b''
    f_wt = L(f, "wt")(x0).squeeze()  # should be b*a'

    a = jnp.array([1.1, 1.2])
    b = jnp.array([[1.3, 1.4], [1.5, 1.6]])
    bp = jnp.array([[[1.3 * 7, 0], [0, 1.6 * 17]], [[0, 1.4 * 11], [1.5 * 13, 0]]])

    ap = jnp.array([[1.1 * 3, 0], [0, 1.2 * 5]])

    assert jnp.allclose(f_t, a)
    assert jnp.allclose(f_w, b)
    assert jnp.allclose(f_ww, jnp.einsum("ai,abj->bij", b, bp))
    assert jnp.allclose(f_wt, jnp.einsum("cj,cd->dj", b, ap))
    # assert jnp.allclose(f_wt, jnp.einsum('a,abj->bj',a,bp) + jnp.einsum('a,abj->bj',sigma,bpp) )


if __name__ == "__main__":
    test_L_operator_logic()
