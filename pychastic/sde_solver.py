import jax
import time
import numpy as np
from pychastic.sde_problem import SDEProblem
from pychastic.wiener import Wiener
from pychastic.wiener import WienerWithZ


class SDESolver:
    '''
    Produces realisations of stochastic process to ``solve`` method.
    Controls numerical integration features via attributes.

    Parameters
    ----------
    adaptive : {true, false}, default: True
         Whether to take fixed-size steps or not.
    scheme : {'euler','milstein'}, default: 'euler'
         Type of scheme used for integration.
    dt : float
         Step size in fixed-step integration.
    min_dt : float
         Minimal value of time step in variable step integrators.
    max_dt : float
         Maximal value of time step in variable step integrators.
    error_terms : int, default: 1
         Number of error terms used for stepsize estimation.
    target_mse_density : float
         Target mean square error density used in variable step integrators.

    '''
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
      elif self.scheme == 'wagner_platen':
          #Kloden-Platen 10.4.1 -- The order 1.5 Strong Taylor Scheme
          def step(x, dt, dw, dz):
              return (
                    x 
                  + problem.a(x)*dt + problem.b(x)*dw
                  + 0.5*problem.b(x)*problem.bp(x)*(dw**2 - dt)
                  + problem.ap(x)*problem.b(x)*dz
                  + 0.5*(problem.a(x)*problem.ap(x) + 0.5*problem.b(x)**2*problem.app(x))*dt**2
                  + (problem.a(x)*problem.bp(x)+0.5*problem.b(x)**2*problem.bpp(x))*(dw*dt-dz)
                  + 0.5*problem.b(x)*(problem.b(x)*problem.bpp(x)+(problem.bp(x))**2)*(1.0/3.0*dw**2-dt)*dw
                  )

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
        '''
        Produce one realisation of the process specified by ``problem``.

        Parameters
        ----------
        problem : SDEProblem
             Stochastic differential equation together with bounary conditions.
        wiener : Wiener, optional
             Underlying Wiener process supplying noise to the equation. Usefull when comparing several solvers or several equations on the same noise.

        Returns
        -------
        dict
            Dictionary containing 3 entries. Under key ``time_values`` a np.array of timestamps on which process was evaluated.
            Under key ``solution_values`` a np.array of stochastic process values at corresponding time instances.
            Under key ``wiener_values`` np.array of values of the underlying Wiener process realisation at coresponding time instances.

        Example
        -------
        >>> problem = pychastic.sde_problem.SDEProblem(lambda x: 1.0,lambda x: -1.0,0.0,0.1)
        >>> solver = pychastic.sde_solver.SDESolver()
        >>> solver.solve(problem)
        {'time_values': array([0.,0.01,...]), 'solution_values' : array([0.,0.0082,...]),'wiener_values' : array([0.,-0.0017,...])} #some values random


        '''
        if self.scheme == 'wagner_platen':
            wiener = wiener or WienerWithZ()
        else:
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
            if self.scheme == 'wagner_platen':
                dz = wiener.get_z(t,t+dt)
                x = step(x, dt, dw)
            else:
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


