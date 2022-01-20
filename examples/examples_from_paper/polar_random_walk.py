import pychastic
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

y_drift = 4.0

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: jnp.array([1 / (2 * x[0]) + y_drift*jnp.sin(x[1]), y_drift*jnp.cos(x[1]) / x[0]]),
    b=lambda x: jnp.array(
        [[jnp.cos(x[1]), jnp.sin(x[1])], [-jnp.sin(x[1]) / x[0], jnp.cos(x[1]) / x[0]]]
    ),
    x0=jnp.array([2.0, 0.0]),
    tmax=1.0,
)

solver = pychastic.sde_solver.SDESolver()
solver.scheme = 'milstein'

dts = [2**(-x) for x in range(1,10,1)]
n_samples = 2**15

theoretical_mean = 1.10714532096375
theoretical_variance = 0.5951002076847987
theoretical_sigma = np.sqrt(theoretical_variance / n_samples)

bias_dict = dict()

for dt in dts:    
    solver.dt = dt
    solution = solver.solve_many(problem,n_samples,seed=0)
    terminal_r = solution['solution_values'][:,-1,0]
    terminal_phi = solution['solution_values'][:,-1,1]
    terminal_phi_canonical = jnp.fmod(terminal_phi + jnp.pi,2*jnp.pi) - jnp.pi # phi in [-pi,pi]
    
    expected_phi = float(jnp.mean(terminal_phi_canonical))
    std_mean_phi = float(jnp.std(terminal_phi_canonical) / jnp.sqrt(n_samples))
    
    measured_bias = expected_phi-theoretical_mean
    sigma_bias = std_mean_phi

    print(f'{int(1/dt)=}')    
    print(f'{measured_bias=}')
    print(f'{sigma_bias=}')
    
    bias_dict[int(1/dt)] = (measured_bias,sigma_bias)
    
for (k,(mu,sigma)) in bias_dict.items():
    print('{'+f'{k},{mu},{sigma}'+'},')
    
