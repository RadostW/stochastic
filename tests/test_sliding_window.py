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
            assert res_vec[i,j] == (test_vec[i+j+1] if i+j+1<dim1 else 0), (i, j)
            
            
    res_mat = pychastic.vectorized_I_generation.sliding_sum(test_mat)
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim2):
                assert res_mat[i,j,k] == (test_mat[i,j+k+1] if j+k+1<dim2 else 0), (i, j, k)
                
                
def test_sliding_abs():
    dim1 = 3
    dim2 = 5

    test_vec = jnp.arange(dim1)
    test_mat = jnp.arange(dim1*dim2).reshape((dim1,dim2))
    
    res_vec = pychastic.vectorized_I_generation.sliding_abs(test_vec)
    
    for i in range(dim1):
        for j in range(dim1):
            assert res_vec[i,j] == (test_vec[abs((i+1)-(j+1))-1] if abs((i+1)-(j+1))-1 in range(dim1) else 0)
            
            
    res_mat = pychastic.vectorized_I_generation.sliding_abs(test_mat)
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim2):
                assert res_mat[i,j,k] == (test_mat[i,abs((j+1)-(k+1))-1] if abs((j+1)-(k+1))-1 in range(dim2) else 0)


def test_make_D_mat():
    noiseterms = 2
    p = 3
    eta = jnp.arange(noiseterms*p).reshape(noiseterms, p)
    zeta = 2*jnp.arange(noiseterms*p).reshape(noiseterms, p)

    D_mat = pychastic.vectorized_I_generation.make_D_mat(eta, zeta)
    D_mat_loopy = pychastic.vectorized_I_generation.make_D_mat_loopy(eta, zeta)

    assert jnp.allclose(D_mat, D_mat_loopy)

if __name__ == "__main__":
    test_make_D_mat()
