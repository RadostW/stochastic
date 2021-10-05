import jax.numpy as jnp

@vmap
def vectorized_fill_diagonal(mat,vec):
    (n,_) = mat.shape
    i, j = jnp.diag_indices(n)
    return mat.at[i,j].set(vec)

# Compare Kloden-Platen (10.3.7), dimension = d, noiseterms = m
# Generate 'steps' stochastic integral increments at once

xi = standard_normal(shape = (steps,1,noiseterms))
dW_scaled = standard_normal(shape = (steps,noiseterms))

mu = standard_normal(size=(steps,1,self.noiseterms))

eta = standard_normal(size=(steps,p,self.noiseterms))

zeta = standard_normal(size=(steps,p,self.noiseterms))

rec = 1/np.arange(1, self.p+1) # 1/r vector
rho = 1/12 - (rec**2).sum()/(2*np.pi**2)

a = math.sqrt(2)*xi+eta

Imat_nodiag = (
    xi*xi.transpose(axes = (0,2,1))/2 
    + np.sqrt(rho)*(
        mu*xi.transpose(axes = (0,2,1)) 
        - xi*mu.transpose(axes = (0,2,1))
        )
    + 1.0 / (2*jnp.pi)*  (
        (
            jnp.expand_dims(a, 2)*jnp.expand_dims(zeta, 3) - jnp.expand_dims(a, 3)*jnp.expand_dims(zeta, 2)
        ).transpose(axes = (0,2,1))*rec
        ).sum(axis=-1)
)

dI_scaled = jnp.vectorized_fill_diagonal(Imat_nodiag, 0.5*(xi**2 - 1)) # Diagonal entries work differently
