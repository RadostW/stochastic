import jax.numpy as jnp

def contract_all(a, b):
  return jnp.tensordot(a, b, axes=len(b.shape))
