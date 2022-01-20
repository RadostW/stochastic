import pychastic.vectorized_I_generation
import pychastic.wiener_integral_moments

import jax # for PRNGKey generation
import jax.numpy as jnp
import numpy as np
import itertools

def test_integral_generation_euler():
    tested_scheme = 'euler'
        
    # Prepare values of means and second moments = E(XY)
    tested_integrals = [[1],[2],[0]] # [dW1,dW2,dt]
    target_means = jnp.array([pychastic.wiener_integral_moments.E(idx)(1) for idx in tested_integrals])
    target_squares = jnp.array([pychastic.wiener_integral_moments.E2(idx,idy)(1) for (idx,idy) in itertools.product(tested_integrals,tested_integrals)])
    
    samples_exponent = 14
    
    seed = 0
    key = jax.random.PRNGKey(seed)
    sample_integrals = pychastic.vectorized_I_generation.get_wiener_integrals(key,scheme=tested_scheme,steps=2**samples_exponent,noise_terms=2)
    
    sample_integrals = jnp.array( [ sample_integrals['d_w'][:,0] , sample_integrals['d_w'][:,1] , jnp.ones_like( sample_integrals['d_w'][:,0] ) ] ).T
    return sample_integrals
    
    sample_means = jnp.mean( sample_integrals , axis = 0)
    
    means_close = jnp.isclose(sample_means,target_means,atol = 5*2**(-samples_exponent/2)) 
    means_error = sample_means - target_means
    assert means_close.all() , 'Expected values incorrect \n' + str({ label : (bool(flag),float(error)) for (flag, error, label) in zip(means_close,means_error,['d_w1','d_w2','d_t'])})
    
    
if __name__ == '__main__':
    test_integral_generation_euler()
    
    
#     tested_integrals = [[1],[2],[0],[1,1],[1,2],[1,0],[2,0],[0,1],[0,2],[1,1,1]]    
