# PyChastic

All python stochastic differential equations (SDE) solver.

Built for stochastic simulations of hydrodynamically interacting 
brownian particles (molecular dynamics), but can do much more 
(such as option pricing in stochastic volitality models).

Uses `jax`, `jax.jit` and `jax.grad` for performace and ease of use.

# Usage

```Python
import pychastic
problem = problem = pychastic.sde_problem.SDEProblem(lambda x: 0.2*x,lambda x: 0.5*x,1.0,2.0)
solver = pychastic.sde_solver.SDESolver()
trajectory = solver.solve(problem)

import matplotlib.pyplot as plt
plt.plot(trajectory['time_values'],trajectory['solution_values'])
plt.show()
```

# License

This software is licensed under MIT license

Copyright (c) Radost Waszkiewicz and Maciej Bartczak (2021).



