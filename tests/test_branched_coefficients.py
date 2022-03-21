import jax.numpy as jnp
import jax


def upper(y):
    return jnp.exp(y)


def lower(y):
    return jnp.exp(2 * y)


def branch_fun(q):
    x = q[0]
    ret = jax.lax.cond(x > 1.0, upper, lower, x)
    return jnp.array([[ret]])


def L_w(f):
    def wrapped(x):
        return jax.jacobian(f)(x) * branch_fun(x)[0,0]

    return wrapped


def solve_many():
    t0 = 0.0
    w0 = jax.numpy.zeros(1)

    id_ = lambda x: x
    f_www = L_w(L_w(L_w(id_)))

    def step(
        x,
        d_www,
    ):

        new_x = (
            # x * d_www[0,0,0] # Ok
            # x * f_www(x)[0,0,0,0] # Ok
            jnp.array([1.0]) * f_www(x)[0, 0, 0, 0] * d_www[0, 0, 0]  # AttributeError
            # x * f_www(x)[0,0,0,0] * d_www[0,0,0] # AttributeError
            # contract_all(f_www(x), d_www) # AttributeError
            # f_www(x)[:,0,0,0] # AttributeError
        )

        return new_x

    key = jax.random.PRNGKey(0)

    def scan_func(x, y):
        xp = step(x, y)
        return (xp, xp)

    def chunk_function(x, y):
        z = jax.lax.scan(scan_func, jnp.array([0.1]), y)[0]
        return z, z

    def get_solution_fragment(starting_state, key):
        wiener_integrals = jax.random.normal(key, shape=(1,2, 1, 1, 1))

        last_state, solution_values = jax.lax.scan(
            chunk_function,
            starting_state,
            wiener_integrals
        )

        return (last_state, last_state)

    @jax.vmap
    def get_solution(key):
        _, chunked_solution = jax.lax.scan(
            lambda state, key: get_solution_fragment(state, key),
            (jnp.array([0.1])),
            jax.random.split(key, 1),
        )

        return chunked_solution

    keys = jax.random.split(key, 1)
    solutions = get_solution(keys)

    return solutions


def test_branched_coefficients():
    trajectories = solve_many()


if __name__ == "__main__":
    trajectories = solve_many()
