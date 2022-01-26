import jax.numpy as jnp
import jax


def fill_diagonal(mat, vec):
    (n, _) = mat.shape
    i, j = jnp.diag_indices(n)
    return mat.at[i, j].set(vec)
vectorized_fill_diagonal = jax.vmap(fill_diagonal, in_axes=(0, 0))


def fill_indices(mat, indices, vec):
    i, j = indices
    return mat.at[i, j].set(vec)
vectorized_fill_indices = jax.vmap(fill_indices, in_axes=(0, 0))

def sliding_sum(vec):
    n = vec.shape[0]
    vec1 = jnp.pad(vec, ((0, n+1),))
    vec2 = jnp.tile(vec1, (n, 1))
    vec3 = vec2.flatten()
    vec4 = vec3[:2*n**2]
    mat = vec4.reshape(n, 2*n)[::-1, n:]
    return mat

sliding_sum = jnp.vectorize(sliding_sum, signature='(k)->(k,k)')

def sliding_abs(vec):
    n = vec.shape[0]
    vec1 = jnp.pad(vec, ((1, n),))
    mat = jnp.tile(vec1, (n, 1)).flatten()[:2*n**2].reshape(n, 2*n)[:, :n]
    mat2 = mat + mat.T
    return mat2

sliding_abs = jnp.vectorize(sliding_abs, signature='(k)->(k,k)')

def double_pad(tensor):
    return jnp.pad(tensor, [(0, 0), (0, tensor.shape[1]), (0, 0)])

def double_pad_vec(vec):
    return jnp.pad(vec, [(0, vec.shape[0])])

# Compare Kloden-Platen (10.3.7), dimension = d, noiseterms = m
# Generate 'steps' stochastic integral increments at once
def get_wiener_integrals(key, steps=1, noise_terms=1, scheme="euler", p=10):
    """
    Calculate moments of principal wiener integrals.

    Parameters
    ----------
    key : jax.PRNGKey
        source of randomness
    steps : int, optional
        number of steps
    noise_terms : int, optional
        wiener process dimension
    scheme : ('euler' or 'milstein' or 'wagner_platen')
        controls order of integrals generated and method of generation
    p : int, optional
        controls series truncation

    Returns
    -------
    dict
        Keys are like 'd_w' or 'd_wt' and values are `jnp.arrays`
    """
    if noise_terms == 1:
        if scheme == 'euler':
            dW_scaled = jax.random.normal(key, shape=(steps, noise_terms))

            return {
                'd_w' : dW_scaled
            }
                
        if scheme == 'milstein':
            dW_scaled = jax.random.normal(key, shape=(steps, noise_terms))
            dI_scaled = 0.5*(dW_scaled**2 - 1)[..., jax.numpy.newaxis] # noise_terms == 1 is special

            return {
                'd_w' : dW_scaled,
                'd_ww' : dI_scaled
            }    
            
        if scheme == 'wagner_platen':
            u = jax.random.normal(key, shape=(2, steps, noise_terms))
            dW_scaled = u[0]
            dZ_scaled = 0.5*(u[0] + 3**(-0.5)*u[1])[..., jax.numpy.newaxis]

            dI_scaled = 0.5*(dW_scaled**2 - 1)[..., jax.numpy.newaxis]

            return {
                'd_w': dW_scaled,
                'd_ww': dI_scaled,
                'd_wt': dZ_scaled,
                'd_tw': dW_scaled[..., jax.numpy.newaxis] - dZ_scaled,
                'd_www': (0.5*((1.0/3.0)*dW_scaled**2-1)*dW_scaled)[..., jax.numpy.newaxis, jax.numpy.newaxis],
            }
        
    # Below multidim implementations
    if scheme == 'euler':
        dW_scaled = jax.random.normal(key, shape=(steps, noise_terms))

        return {
            'd_w' : dW_scaled
        }
    elif scheme == 'milstein':
        key1, key2, key3, key4 = jax.random.split(key, num=4)
        xi = jax.random.normal(key1, shape=(steps, 1,noise_terms))
        dW_scaled = xi.squeeze() #jax.random.normal(key2, shape=(steps,noise_terms))

        mu = jax.random.normal(key2, shape=(steps, 1,noise_terms))

        eta = jax.random.normal(key3, shape=(steps, p,noise_terms))

        zeta = jax.random.normal(key4, shape=(steps, p,noise_terms))

        rec = 1 / jax.numpy.arange(1, p + 1)  # 1/r vector
        rho = 1 / 12 - (rec ** 2).sum() / (2 * jax.numpy.pi ** 2)

        a = jax.numpy.sqrt(2) * xi + eta

        Imat_nodiag = (
            xi * jax.numpy.transpose(xi, axes=(0, 2, 1)) / 2
            + jax.numpy.sqrt(rho)
            * (mu * jax.numpy.transpose(xi, axes=(0, 2, 1)) - xi * jax.numpy.transpose(mu, axes=(0, 2, 1)))
            + 1.0
            / (2 * jnp.pi)
            * (
                (
                    jnp.expand_dims(a, 2) * jnp.expand_dims(zeta, 3)
                    - jnp.expand_dims(a, 3) * jnp.expand_dims(zeta, 2)
                )
                * rec[:, jax.numpy.newaxis, jax.numpy.newaxis]
            ).sum(axis=1)
        )

        dI_scaled = vectorized_fill_diagonal(
            Imat_nodiag, 0.5 * (xi ** 2 - 1).squeeze()
        )  # Diagonal entries work differently

        return {
                'd_w': dW_scaled,
                'd_ww': dI_scaled
            }
    elif scheme == 'wagner_platen':
        #raise NotImplementedError # WIP
        
        key1, key2, key3, key4, key5 = jax.random.split(key, num=5)
        
        # Notation of Kloeden-Platen (10.4.7) and onwards
        xi = jax.random.normal(key1, shape=(steps, noise_terms))
        zeta = jax.random.normal(key2, shape=(steps, p, noise_terms))
        eta = jax.random.normal(key3, shape=(steps, p, noise_terms))
        mu = jax.random.normal(key4, shape=(steps, noise_terms))
        phi = jax.random.normal(key5, shape=(steps, noise_terms))
        
        rec = 1.0/jnp.arange(1,p+1)[jnp.newaxis,:] # 1/r vector
        
        r_vec = jnp.arange(1,p+1)
        C_mat_coeffs = (r_vec[:,jnp.newaxis]/((r_vec[:,jnp.newaxis]**2 - r_vec[jnp.newaxis,:]**2)+jnp.eye(p)))*(1 - jnp.eye(p))[jnp.newaxis,:,:] # 1(r!=l) * r / (r**2 - l**2)
        
        rho = (1.0/ 12.0) - (rec ** 2).sum() / (2 * jax.numpy.pi ** 2)
        alpha = (jnp.pi**2 / 180.0) - (0.5 / jnp.pi**2) * (rec**4).sum()
        
        a_vec = (- jnp.sqrt(2)/jnp.pi* jnp.sum( rec[:,:,jnp.newaxis]*zeta , axis = 1) - 2*jnp.sqrt(rho)*mu )
        b_vec = (1 / jnp.sqrt(2) ) *  jnp.sum( (rec[:,:,jnp.newaxis]**2) * eta  , axis = 1) + jnp.sqrt(alpha)*phi
        
        A_mat = (0.5 / jnp.pi)*jnp.sum(
            rec[:,:,jnp.newaxis,jnp.newaxis] * (
                zeta[:,:,:,jnp.newaxis] * eta[:,:,jnp.newaxis,:]
                - eta[:,:,:,jnp.newaxis] * zeta[:,:,jnp.newaxis,:]
                ) ,
            axis=1
        )
        B_mat = (1.0 / (4.0 * jnp.pi**2)) * jnp.sum( (rec[:,:,jnp.newaxis,jnp.newaxis]**2) * (
                zeta[:,:,:,jnp.newaxis] * zeta[:,:,jnp.newaxis,:] 
                + eta[:,:,:,jnp.newaxis] * eta[:,:,jnp.newaxis,:]
            ) , axis = 1)
        C_mat = -(1.0 / (2.0 * jnp.pi**2)) * jnp.sum( C_mat_coeffs[:,:,:,jnp.newaxis,jnp.newaxis] * (
                zeta[:,:,jnp.newaxis,:,jnp.newaxis] * zeta[:,jnp.newaxis,:,jnp.newaxis,:] * rec[:,jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
                - eta[:,:,jnp.newaxis,:,jnp.newaxis] * eta[:,jnp.newaxis,:,jnp.newaxis,:] * rec[:,:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
            ) , axis = (1,2) )
            
        def sliding_abs_alt(ten):
            return jnp.transpose(sliding_abs(jnp.transpose(ten,axes = (0,2,1))), axes = (0,2,3,1))
            
        def sliding_sum_alt(ten):
            return jnp.transpose(sliding_sum(jnp.transpose(ten,axes = (0,2,1))), axes = (0,2,3,1))
       
        r_vec2 = r_vec[:,jnp.newaxis]
        l_vec2 = r_vec[jnp.newaxis,:]
       
        D_mat_coeffs =  ((1.0 / ( double_pad_vec(l_vec2) * sliding_abs( r_vec ) + jnp.eye(p) )) * (1 - jnp.eye(p)) )[jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis,jnp.newaxis] # 1 / r (r-l)
        D_mat_coeffs2 = (1.0 / ( double_pad_vec(r_vec2) * sliding_sum( r_vec ) ))[jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis,jnp.newaxis] # 1 / l (r+l)
        sgn_coeffs =  (jnp.tri(p).T-jnp.tri(p))[jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis,jnp.newaxis] # sgn(l-r)
            
        D_mat = jnp.zeros((noise_terms,noise_terms,noise_terms)) = (1.0 / (jnp.pi**2 * 2.0**(5/2)) * jnp.sum(
            D_mat_coeffs * ( zeta[:,:,jnp.newaxis,jnp.newaxis,:,jnp.newaxis] * (
                    double_pad(zeta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_abs_alt(eta)[:,:,:,jnp.newaxis,jnp.newaxis,:] * sgn_coeffs +
                    double_pad(eta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_abs_alt(zeta)[:,:,:,jnp.newaxis,jnp.newaxis,:]
                )
                +
                eta[:,:,jnp.newaxis,jnp.newaxis,:,jnp.newaxis] * (
                    - double_pad(zeta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_abs_alt(zeta)[:,:,:,jnp.newaxis,jnp.newaxis,:] * sgn_coeffs +
                    double_pad(eta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_abs_alt(eta)[:,:,:,jnp.newaxis,jnp.newaxis,:]
                )
                )
        axis = (1,2))
        +
        jnp.sum(
            D_mat_coeffs2 * ( zeta[:,:,jnp.newaxis,jnp.newaxis,:,jnp.newaxis] * (
                   - double_pad(zeta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_sum_alt(eta)[:,:,:,jnp.newaxis,jnp.newaxis,:] +
                    double_pad(eta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_sum_alt(zeta)[:,:,:,jnp.newaxis,jnp.newaxis,:]
                )
                +
                eta[:,:,jnp.newaxis,jnp.newaxis,:,jnp.newaxis] * (
                    double_pad(zeta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_sum_alt(zeta)[:,:,:,jnp.newaxis,jnp.newaxis,:] +
                    double_pad(eta)[:,jnp.newaxis,:,:,jnp.newaxis,jnp.newaxis] * sliding_sum_alt(eta)[:,:,:,jnp.newaxis,jnp.newaxis,:]
                )
            )                
            )    
        , axis = (1,2))
        )


        
        dW_scaled = xi.squeeze()
        dWT_scaled = 0.5*(xi+a_vec)[:,:,jnp.newaxis] # time axes have dim=1
        dTW_scaled = dW_scaled[:,jnp.newaxis,:] - jnp.transpose(dWT_scaled, axes = (0,2,1))
        
        #J_mat = NotImplementedError # check Kloeden-Platen (5.8.11)
        J_mat = jnp.zeros((steps,noise_terms,noise_terms,noise_terms)) # TODO:NotImplementedError
        
        dWW_diag_scaled = 0.5*(dW_scaled**2-1.0) # only diagonal elements
        dWWW_diag_scaled = J_mat
        
        dWW_nodiag_scaled = (
                0.5 * xi[:,:,jnp.newaxis] * xi[:,jnp.newaxis,:]
                - 0.5 * (
                      xi[:,:,jnp.newaxis] * a_vec[:,jnp.newaxis,:]
                    - xi[:,jnp.newaxis,:] * a_vec[:,:,jnp.newaxis])
                + A_mat)
                
        
        dWW_scaled = vectorized_fill_diagonal(dWW_nodiag_scaled , dWW_diag_scaled)
        
        dWWW_scaled = jnp.zeros_like(J_mat)
        
        return {
                'd_w': dW_scaled,
                'd_ww': dWW_scaled,
                'd_wt': dWT_scaled,
                'd_tw': dTW_scaled,
                'd_www': dWWW_scaled,
            }
        
    else:
        raise NotImplementedError

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    get_wiener_integrals(key, steps=3, noise_terms=1)
    get_wiener_integrals(key, steps=3, noise_terms=2)
