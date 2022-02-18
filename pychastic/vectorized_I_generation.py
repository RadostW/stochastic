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


def make_C_mat(eta, zeta):
    assert eta.ndim == 2
    assert eta.shape == zeta.shape
    m, p = eta.shape

    l = jnp.arange(1, p+1).reshape(p, 1)
    r = jnp.arange(1, p+1).reshape(1, p)

    l_big = l.reshape(1, 1, p, 1)
    r_big = l.reshape(1, 1, 1, p)

    summands = (l != r) * r/(r**2-l**2+jnp.eye(p)) * (
        (1/l_big) * zeta[:, r-1].reshape(m, 1, 1, p)*zeta[:, l-1].reshape(1, m, p, 1) +
        (1/r_big) *  eta[:, r-1].reshape(m, 1, 1, p)* eta[:, l-1].reshape(1, m, p, 1)
    )

    C_mat = -1/(2*jnp.pi**2)*summands.sum(axis=(-2, -1))

    return C_mat

vectorized_make_C_mat = jax.vmap(make_C_mat)

def make_D_mat_loopy(eta, zeta):
    assert eta.ndim == 2
    assert eta.shape == zeta.shape
    noiseterms, p = eta.shape

    def take(tensor, j, idx):
        if idx > p:
            return 0
        else:
            return tensor[j, idx-1]

    D_mat = jnp.zeros((noiseterms, noiseterms, noiseterms))

    for j1 in range(noiseterms):
        for j2 in range(noiseterms):
            for j3 in range(noiseterms):
                # first sum
                for r in range(1, p+1):
                    for l in range(1, p+1):
                        D_mat = D_mat.at[j1, j2, j3].add( -1/(r*(l+r))*(
                            take(zeta, j2, l)*(take(zeta, j3, l+r)*take(eta, j1, r) - take(zeta, j3, r)*take(eta, j1, l+r)) +
                            take(eta, j2, l)*(take(zeta, j1, r)*take(zeta, j3, l+r) + take(eta, j1, r)*take(eta, j3, l+r))
                        ))

                # second sum
                for l in range(1, p+1):
                    for r in range(1, l-1+1):
                        D_mat = D_mat.at[j1, j2, j3].add( 1/(r*(l-r))*(
                            take(zeta, j2, l)*(take(zeta, j1, r)*take(eta, j3, l-r) + take(zeta, j3, l-r)*take(eta, j1, r)) -
                            take(eta, j2, l)*(take(zeta, j1, r)*take(zeta, j3, l-r) - take(eta, j1, r)*take(eta, j3, l-r))
                        ))

                # third sum
                for l in range(1, p+1):
                    for r in range(l+1, 2*p+1):
                        D_mat = D_mat.at[j1, j2, j3].add( 1/(r*(r-l))*(
                            take(zeta, j2, l)*(take(zeta, j3, r-l)*take(eta, j1, r) - take(zeta, j1, r)*take(eta, j3, r-l)) +
                            take(eta, j2, l)*(take(zeta, j1, r)*take(zeta, j3, r-l) + take(eta, j1, r)*take(eta, j3, r-l))
                        ))

    D_mat *= 1/(jnp.pi**2*2**(5/2))
    return D_mat

def make_D_mat(eta, zeta):
    assert eta.ndim == 2
    assert eta.shape == zeta.shape
    m, p = jnp.shape(eta)

    def take(tensor, idx, fill = 0):
        # Non jit-friendly implementation
        # illegal = jnp.logical_or(idx > p,idx < 1)
        # return tensor[..., idx-1].at[..., illegal].set(fill)
        legalized_idx = jnp.clip(idx,a_min = 1,a_max = p)
        illegal_mask = jnp.logical_or(idx > p,idx < 1)
        return tensor[...,legalized_idx-1]*(1 - 1*illegal_mask) + illegal_mask*fill

    # first term
    # summands shape: (m, m, m, p, p)

    l = jnp.arange(1, p+1).reshape(p, 1)
    r = jnp.arange(1, p+1).reshape(1, p)

    summands_sum = 1/(r*(l+r))*(
        zeta.reshape(1, m, 1, p, 1)*(
            take(zeta, l+r).reshape(1, 1, m, p, p) * eta.reshape(m, 1, 1, 1, p)
            - zeta.reshape(1, 1, m, 1, p) * take(eta, l+r).reshape(m, 1, 1, p, p)
        )
        + eta.reshape(1, m, 1, p, 1)*(
            zeta.reshape(m, 1, 1, 1, p) * take(zeta, l+r).reshape(1, 1, m, p, p)
            + eta.reshape(m, 1, 1, 1, p) * take(eta, l+r).reshape(1, 1, m, p, p)
        )
    )
    D_mat_sum = summands_sum.sum(axis=(-2, -1))

    # second term
    # summands shape: (m, m, m, p, 2p)

    l = jnp.arange(1, p+1).reshape(p, 1)
    r = jnp.arange(1, 2*p+1).reshape(1, 2*p) # note rectangular sum

    summands_diff_lower = (r<l) * 1.0/(
        r*take(r,l-r,fill=1.0)
        )*(
        zeta.reshape(1, m, 1, p, 1)*(
            take(zeta,r).reshape(m, 1, 1, 1, 2*p) * take(eta, l-r).reshape(1, 1, m, p, 2*p)
            + take(eta,r).reshape(m, 1, 1, 1, 2*p) * take(zeta, l-r).reshape(1, 1, m, p, 2*p) 

        )
        - eta.reshape(1, m, 1, p, 1)*(
            take(zeta,r).reshape(m, 1, 1, 1, 2*p) * take(zeta, l-r).reshape(1, 1, m, p, 2*p) 
            - take(eta,r).reshape(m, 1, 1, 1, 2*p) * take(eta, l-r).reshape(1, 1, m, p, 2*p)
        )
    )
    D_mat_diff_lower = summands_diff_lower.sum(axis=(-2, -1))


    # third term
    # summands shape: (m, m, m, p, 2p)

    summands_diff_upper = (r>l) * 1.0/(
        r*take(r,r-l,fill=1.0)
        )*(
        zeta.reshape(1, m, 1, p, 1)*(
            take(zeta,r).reshape(m, 1, 1, 1, 2*p) * take(eta, r-l).reshape(1, 1, m, p, 2*p)
            - take(eta,r).reshape(m, 1, 1, 1, 2*p) * take(zeta, r-l).reshape(1, 1, m, p, 2*p) 

        )
        + eta.reshape(1, m, 1, p, 1)*(
            take(zeta,r).reshape(m, 1, 1, 1, 2*p) * take(zeta, r-l).reshape(1, 1, m, p, 2*p) 
            + take(eta,r).reshape(m, 1, 1, 1, 2*p) * take(eta, r-l).reshape(1, 1, m, p, 2*p)
        )
    )
    D_mat_diff_upper = summands_diff_upper.sum(axis=(-2, -1))


    D_mat = 1/(jnp.pi**2*2**(5/2)) * (-D_mat_sum + D_mat_diff_lower + D_mat_diff_upper)
    return D_mat

vectorized_make_D_mat = jax.vmap(make_D_mat)

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

        key1, key2, key3, key4, key5 = jax.random.split(key, num=5)

        # Notation of Kloeden-Platen (10.4.7) and onwards
        xi = jax.random.normal(key1, shape=(steps, noise_terms))
        zeta = jax.random.normal(key2, shape=(steps, p, noise_terms))
        eta = jax.random.normal(key3, shape=(steps, p, noise_terms))
        mu = jax.random.normal(key4, shape=(steps, noise_terms))
        phi = jax.random.normal(key5, shape=(steps, noise_terms))

        rec = 1.0/jnp.arange(1,p+1)[jnp.newaxis,:] # 1/r vector

        rho = (1.0/ 12.0) - (rec ** 2).sum() / (2 * jax.numpy.pi ** 2)
        alpha = (jnp.pi**2 / 180.0) - (0.5 / jnp.pi**2) * (rec**4).sum()

        a_vec = (- jnp.sqrt(2)/jnp.pi* jnp.sum( rec[:,:,jnp.newaxis]*zeta , axis = 1) - 2*jnp.sqrt(rho)*mu )
        b_vec = 1 / ( jnp.sqrt(2)*jnp.pi ) *  jnp.sum( (rec[:,:,jnp.newaxis]**2) * eta  , axis = 1) + jnp.sqrt(alpha)*phi

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

        C_mat = vectorized_make_C_mat(eta.transpose((0, 2, 1)), zeta.transpose((0, 2, 1)))

        D_mat = vectorized_make_D_mat(eta.transpose((0, 2, 1)), zeta.transpose((0, 2, 1)))

        dW_scaled = xi.squeeze()
        dWT_scaled = 0.5*(xi+a_vec)[:,:,jnp.newaxis] # time axes have dim=1
        dTW_scaled = dW_scaled[:,jnp.newaxis,:] - jnp.transpose(dWT_scaled, axes = (0,2,1))


        dWW_diag_scaled = 0.5*(dW_scaled**2-1.0) # only diagonal elements

        dWW_nodiag_scaled = (
                0.5 * xi[:,:,jnp.newaxis] * xi[:,jnp.newaxis,:]
                - 0.5 * (
                      xi[:,:,jnp.newaxis] * a_vec[:,jnp.newaxis,:]
                    - xi[:,jnp.newaxis,:] * a_vec[:,:,jnp.newaxis])
                + A_mat)


        dWW_scaled = vectorized_fill_diagonal(dWW_nodiag_scaled , dWW_diag_scaled)
        
        m = noise_terms
        J_tww = (
            (1.0 / 6.0) * xi.reshape(steps, m, 1) * xi.reshape(steps, 1, m) 
            - (1.0 / jnp.pi) * xi.reshape(steps, 1, m) * b_vec.reshape(steps, m, 1) 
            + B_mat # .T?
            - 0.25 * a_vec.reshape(steps, 1, m) * xi.reshape(steps, m, 1)
            + (0.5 / jnp.pi) * xi.reshape(steps, m, 1) * b_vec.reshape(steps, 1, m)
            + C_mat
            + 0.5 * A_mat
        )
        
        J_tw = 1/2 * (xi - a_vec)
        J_wtw = (
            1/6 * xi.reshape(steps, m, 1) * xi.reshape(steps, 1, m)
            + 1/2 * a_vec.reshape(steps, m, 1) * J_tw.reshape(steps, 1, m)
            + 1/(2*jnp.pi) * xi.reshape(steps, 1, m) * b_vec.reshape(steps, m, 1)
            - B_mat
            - 1/4 * a_vec.reshape(steps, 1, m) * xi.reshape(steps, m, 1)
            + 1/(2*jnp.pi) * xi.reshape(steps, m, 1) * b_vec.reshape(steps, 1, m)
        )
        
        J_ww = (
                0.5 * xi[:,:,jnp.newaxis] * xi[:,jnp.newaxis,:]
                - 0.5 * (
                      xi[:,:,jnp.newaxis] * a_vec[:,jnp.newaxis,:]
                    - xi[:,jnp.newaxis,:] * a_vec[:,:,jnp.newaxis])
                + A_mat)
        J_mat = (
            xi.reshape(steps, m, 1, 1) * J_tww.reshape(steps, 1, m, m)
            + 0.5 * a_vec.reshape(steps, m, 1, 1) * J_ww.reshape(steps, 1, m, m)
            + (0.5 / jnp.pi) * b_vec.reshape(steps, m, 1, 1) * xi.reshape(steps, 1, m, 1) * xi.reshape(steps, 1, 1, m)
            - xi.reshape(steps, 1, m, 1) * B_mat.reshape(steps, m, 1, m)
            + xi.reshape(steps, 1, 1, m) * (0.5 * A_mat.reshape(steps, m, m, 1) - jnp.transpose(C_mat, axes = (0,2,1)).reshape(steps, m, m, 1))
            + D_mat
        )
        
        dWWW_scaled = J_mat - 0.5 * (
            jnp.eye(m).reshape(1, m, m, 1) * dTW_scaled.reshape(steps, 1, 1, m) +
            jnp.eye(m).reshape(1, 1, m, m) * dWT_scaled.reshape(steps, m, 1, 1)
        )
        
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
