import jax.numpy as jnp
import jax

def get_wiener_integrals(key, steps=1, noise_terms=1, scheme="euler", p=10):

    u = jax.random.normal(key, shape=(2, steps, noise_terms))
    dW_scaled = u[0]
    dI_scaled = 0.5 * (dW_scaled ** 2 - 1)[:,:, jax.numpy.newaxis]

    return {
        "d_w": jnp.zeros((steps,1)),
        "d_www": dW_scaled[:,:, jax.numpy.newaxis, jax.numpy.newaxis],
    }
