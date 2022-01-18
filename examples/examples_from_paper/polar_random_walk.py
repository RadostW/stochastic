import pychastic
import jax.numpy as jnp
import numpy as np

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: jnp.array([1 / (2 * x[0]), 0]),
    b=lambda x: jnp.array(
        [[jnp.cos(x[1]), jnp.sin(x[1])], [-jnp.sin(x[1]) / x[0], jnp.cos(x[1]) / x[0]]]
    ),
    x0=jnp.array([3.0, 0.0]),
    tmax=1.0,
)

solver = pychastic.sde_solver.SDESolver()
solver.scheme = 'euler'

dts = [2**(-x) for x in range(0,10,2)]
n_samples = 2**25

theoretical_mean = 3.172577287900717
theoretical_variance = 0.934753
theoretical_sigma = np.sqrt(theoretical_variance / n_samples)

bias_dict = dict()

for dt in dts:    
    solver.dt = dt
    solution = solver.solve_many(problem,n_samples,seed=0)
    terminal_r = solution['solution_values'][:,-1,0]
    terminal_phi = solution['solution_values'][:,-1,1]
    expected_r = float(jnp.mean(terminal_r))
    std_mean_r = float(jnp.std(terminal_r) / jnp.sqrt(n_samples))
    
    true_r = jnp.sum((solution['wiener_values'][:,-1]+problem.x0)**2,axis=1)**0.5
    print(f'{expected_r - theoretical_mean}')
    #print(f'{true_r=}')
    #print(f'{terminal_r=}')
    #print(f'error={terminal_r-true_r}')
    #print(f'max_error={jnp.max(terminal_r-true_r)}')
    #print(f'mean_error={jnp.mean(terminal_r-true_r)}')
