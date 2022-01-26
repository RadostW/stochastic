import pychastic.vectorized_I_generation

import jax.numpy as jnp

def test_sliding_sum():
    dim1 = 3
    dim2 = 5

    test_vec = jnp.arange(dim1)
    test_mat = jnp.arange(dim1*dim2).reshape((dim1,dim2))
    
    res_vec = pychastic.vectorized_I_generation.sliding_sum(test_vec)
    
    for i in range(dim1):
        for j in range(dim1):
            assert res_vec[i,j] = test_vec[i+j]
            
            
    res_mat = pychastic.vectorized_I_generation.sliding_sum(test_vec)
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim2):
                assert res_mat[i,j,k] = test_mat[i,j+k]
