from pychastic.problems.kp_4_27 import KloedenPlaten4_27
from pychastic.sde_solver import SDESolver

problem = KloedenPlaten4_27
solver = SDESolver()
solver.dt = 0.01
solution = solver.solve(problem)

exact = problem.exact_solution(problem.x0, solution['time_values'], solution['wiener_values'])

import matplotlib.pyplot as plt
plt.plot(solution['time_values'], solution['solution_values'], marker='x', label='simulated')
plt.plot(solution['time_values'], exact, label='true')
plt.legend()
#plt.savefig('sol.png')
#plt.close()
plt.show()

#plt.plot(solution['time_values'], solution['solution_values']-exact)
#plt.title('Error')
#plt.savefig('error.png')
#plt.close()
