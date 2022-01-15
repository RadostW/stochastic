import pychastic
import jax.numpy as jnp
import numpy as np

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: jnp.array([1 / (2 * x[0]), 0]),
    b=lambda x: jnp.array(
        [[jnp.cos(x[1]), jnp.sin(x[1])], [-jnp.sin(x[1]) / x[0], jnp.cos(x[1]) / x[0]]]
    ),
    x0=jnp.array([1.0, 0.0]),
    tmax=1.0,
)

solver = pychastic.sde_solver.SDESolver()
solver.scheme = 'euler'

dts = [2**(-x) for x in range(12,18)]
n_samples = 2**10

theoretical_mean = 1.548572460551145
theoretical_variance = 0.7296709882254283
theoretical_sigma = np.sqrt(theoretical_variance / n_samples)

bias_dict = dict()

for dt in dts:
    expected_r = np.zeros(2**6)
    for i, _ in enumerate(expected_r):
        solver.dt = dt
        solution = solver.solve_many(problem,n_samples,seed=i)
        terminal_r = solution['solution_values'][:,-1,0]
        terminal_phi = solution['solution_values'][:,-1,1]
        expected_r[i] = float(jnp.mean(terminal_r))
        std_mean_r = float(jnp.std(terminal_r) / jnp.sqrt(n_samples))
        print(f'{expected_r=}')
        print(f'{std_mean_r=}')
        print(f'{theoretical_mean=}')
        print(f'{theoretical_sigma=}')
        print(f'bias={expected_r[i]-theoretical_mean}')
    print(f'bias={np.mean(expected_r)-theoretical_mean}')    
    bias_dict[dt]=np.mean(expected_r)-theoretical_mean
    
for k,v in bias_dict.items():
    print('{'+f'{k},{v}'+'},')

