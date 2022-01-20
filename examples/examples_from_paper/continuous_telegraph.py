import pychastic
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: -x*(1-x)*(1+x),
    b=lambda x: (1-x)*(1+x),
    x0=jnp.array(15.0/16.0),
    tmax=1.0/4.0,
)

solver = pychastic.sde_solver.SDESolver()
solver.scheme = 'milstein'

print(f'SOLVER SCHEME {solver.scheme}')

dts = [2**(-x) for x in range(3,10,1)]
n_samples = 2**20

theoretical_mean = 0.905450297089824
theoretical_variance = 0.01000973989252807
theoretical_sigma = np.sqrt(theoretical_variance / n_samples)

bias_dict = dict()

for dt in dts:    
    solver.dt = dt
    solution = solver.solve_many(problem,n_samples,seed=0)
    terminal_x = solution['solution_values'][:,-1,0]
    
    expected_x = float(jnp.mean(terminal_x))
    std_mean_x = float(jnp.std(terminal_x) / jnp.sqrt(n_samples))
    
    measured_bias = expected_x-theoretical_mean
    sigma_bias = std_mean_x

    true_x = jnp.tanh( solution['wiener_values'][:,-1,0] + jnp.arctanh(problem.x0))
    print(f'{jnp.mean(jnp.abs(terminal_x - true_x))=}')
    print(f'{jnp.mean(true_x)-theoretical_mean=}')

    print(f'{int(1/dt)=}')    
    print(f'{measured_bias=}')
    print(f'{sigma_bias=}')
    print(f'{measured_bias/sigma_bias=}')
    
    bias_dict[int(1/dt)] = (measured_bias,sigma_bias)
    
for (k,(mu,sigma)) in bias_dict.items():
    print('{'+f'{k},{mu},{sigma}'+'},')
    
