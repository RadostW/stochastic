import jax
import time
import numpy as np
from .random_ import normal
from .sde_problem import SDEProblem
from paczka.wiener import Wiener

class SDESolver:
    def __init__(self):
        self.adaptive = False
        self.scheme = 'euler'  # euler | milstein
        self.dt = 0.01
        self.min_dt = self.dt/10
        self.max_dt = self.dt*10
        self.seed = None
        self.error_terms = 1
        self.target_mse_density = 0.1

    def get_step_function(self, problem):
      if self.scheme == 'euler':
          def step(x, dt, dw):
              return x + problem.a(x)*dt + problem.b(x)*dw
      elif self.scheme == 'milstein':
          def step(x, dt, dw):
              return x + problem.a(x)*dt + problem.b(x)*dw + 0.5*problem.b(x)*problem.bp(x)*(dw**2 - dt)
      else:
        raise KeyError('wrong scheme')

      step = jax.jit(step)
      step(1.0, 1.0, 1.0)  # to compile
      return step

    def get_optimal_dt_function(self, problem):
      if self.scheme == 'euler':
          if self.error_terms == 1:
            def optimal_dt(x):
              return self.target_mse_density/problem.a(x)**2
          
          elif self.error_terms == 2:
            def optimal_dt(x):
              alfa = problem.a(x)**2
              beta = (
                  (problem.a(x)*problem.bp(x) + problem.b(x)**2*problem.bpp(x)/2)**2/3
                  + (problem.ap(x)*problem.b(x))**2/3
                  + (problem.b(x)*(problem.bp(x)**2 + problem.b(x)*problem.bpp(x)))**2/6
              )
              return 2*self.target_mse_density/(jax.numpy.sqrt(alfa**2+4*beta*self.target_mse_density)+alfa)
          
          else:
            raise ValueError
              
      elif self.scheme == 'milstein':
          if self.error_terms == 1:
            def optimal_dt(x):
              beta = (
                  (problem.a(x)*problem.bp(x) + problem.b(x)**2*problem.bpp(x)/2)**2/3
                  + (problem.ap(x)*problem.b(x))**2/3
                  + (problem.b(x)*(problem.bp(x)**2 + problem.b(x)*problem.bpp(x)))**2/6
              )
              return jax.numpy.sqrt(self.target_mse_density/beta)
          else:
            raise ValueError
      else:
          raise ValueError

      optimal_dt = jax.jit(optimal_dt)
      optimal_dt(1.0)  # to complie
      return optimal_dt        
    
    def solve(self, problem: SDEProblem, wiener: Wiener = None):
        wiener = wiener or Wiener()
        step = self.get_step_function(problem)
        if self.adaptive:
          optimal_dt = self.get_optimal_dt_function(problem)
        
        # initialize values
        dt = self.dt
        t = 0.0
        x = problem.x0
        w = 0.0

        # initialize "trajectories"
        time_values = [0.0]
        solution_values = [x]
        wiener_values = [0.0]
                
        t0 = time.time()
        while True:
            # adapt step
            if self.adaptive:
              dt = optimal_dt(x)
              dt = max(dt, self.min_dt)
              dt = min(dt, self.max_dt)

            t += dt
            dw = wiener.get_w(t+dt) - wiener.get_w(t)
            w += dw
            x = step(x, dt, dw)

            # update "trajectories"
            solution_values.append(x)            
            wiener_values.append(w)
            time_values.append(t)

            if t >= problem.tmax:
              break
        print((time.time()-t0)*1000)
        return dict(
            time_values=np.array(time_values),
            solution_values=np.array(solution_values),
            wiener_values=np.array(wiener_values),
        )

