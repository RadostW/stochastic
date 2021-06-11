import jax.numpy as jnp

def contract(a, b):
  return jnp.tensordot(a, b, axes=len(b.shape))
