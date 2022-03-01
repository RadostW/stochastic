import pychastic.vectorized_I_generation

import jax.numpy as jnp

def test_make_D_mat():
    noiseterms = 2
    p = 3
    
    eta = 0*jnp.arange(noiseterms*p).reshape(noiseterms, p)
    zeta = 2*jnp.arange(noiseterms*p).reshape(noiseterms, p)

    D_mat = pychastic.vectorized_I_generation.make_D_mat(eta, zeta)
    D_mat_loopy = pychastic.vectorized_I_generation.make_D_mat_loopy(eta, zeta)

    assert jnp.allclose(D_mat, D_mat_loopy)
    
    
    eta = 2*jnp.arange(noiseterms*p).reshape(noiseterms, p)
    zeta = 0*jnp.arange(noiseterms*p).reshape(noiseterms, p)

    D_mat = pychastic.vectorized_I_generation.make_D_mat(eta, zeta)
    D_mat_loopy = pychastic.vectorized_I_generation.make_D_mat_loopy(eta, zeta)

    assert jnp.allclose(D_mat, D_mat_loopy)
    
    
    eta = 3*jnp.arange(noiseterms*p).reshape(noiseterms, p)
    zeta = 5*jnp.arange(noiseterms*p).reshape(noiseterms, p)

    D_mat = pychastic.vectorized_I_generation.make_D_mat(eta, zeta)
    D_mat_loopy = pychastic.vectorized_I_generation.make_D_mat_loopy(eta, zeta)

    assert jnp.allclose(D_mat, D_mat_loopy)

def test_vectorized_make_D_mat():
    noiseterms = 2
    p = 3
    steps = 4
    
    eta = 3*jnp.arange(steps*noiseterms*p).reshape(steps, noiseterms, p)
    zeta = 5*jnp.arange(steps*noiseterms*p).reshape(steps, noiseterms, p)

    vectorized_result = pychastic.vectorized_I_generation.vectorized_make_D_mat(eta, zeta)

    for j in range(steps):
        assert jnp.allclose(
            vectorized_result[j],
            pychastic.vectorized_I_generation.make_D_mat(eta[j], zeta[j])    
        )

if __name__ == "__main__":
    test_make_D_mat()
