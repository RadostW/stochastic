import jax
import time
#import jax.numpy as np
import numpy as np
from pychastic.sde_problem import SDEProblem
from pychastic.sde_problem import VectorSDEProblem
from pychastic.wiener import VectorWienerWithI, Wiener
from pychastic.wiener import WienerWithZ
from pychastic.wiener import VectorWiener


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
    def __init__(self, adaptive = False, scheme = 'euler', dt = 0.01, min_dt = 0.000,
                 max_dt = 0.01, seed = None, error_terms = 1,target_mse_density = 0.0000001):
        self.adaptive = adaptive
        self.scheme = scheme  # euler | milstein | wagner_platen | adaptive_euler | adaptive_milstein
        if scheme == 'adaptive_euler':
            self.scheme = 'euler'
            self.adaptive = True
        if scheme == 'adaptive_milstein':
            self.scheme = 'milstein'
            self.adaptive = True
        self.dt = dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.seed = seed
        self.error_terms = error_terms
        self.target_mse_density = target_mse_density

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
      if self.scheme == 'wagner_platen':
            step(1.0, 1.0, 1.0, 1.0)  # to compile
      else:
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
          min_dt = self.dt/self.dt_adapting_factor
          max_dt = self.dt*self.dt_adapting_factor
        
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
              dt = optimal_dt(x).item()
              dt = max(dt, self.min_dt)
              dt = min(dt, self.max_dt)
              # print(f'x = {x}, t = {t}, dt = {dt}')

            t += dt
            dw = wiener.get_w(t+dt) - wiener.get_w(t)
            w += dw
            if self.scheme == 'wagner_platen':
                dz = wiener.get_z(t,t+dt)
                print((x, dt, dw, dz))
                x = step(x, dt, dw, dz)
            else:
                x = step(x, dt, dw)

            # update "trajectories"
            solution_values.append(x)            
            wiener_values.append(w)
            time_values.append(t)

            if t >= problem.tmax:
              break
        main_loop_time_ms = (time.time()-t0)*1000

        return dict(
            time_values=np.array(time_values),
            solution_values=np.array(solution_values),
            wiener_values=np.array(wiener_values),
            main_loop_time_ms = main_loop_time_ms
        )

class VectorSDESolver:
    '''
    Produces realisations of stochastic process to ``solve`` method.
    Controls numerical integration features via attributes.

    Parameters
    ----------
    scheme : {'euler', 'commutative_milstein', 'milstein'}, default: 'euler'
         Type of scheme used for integration.
    dt : float
         Step size in fixed-step integration.

    '''
    def __init__(self, scheme = 'euler', dt = 0.01):
        self.scheme = scheme  # euler | milstein
        self.dt = dt

    def get_step_function(self, problem):
        if self.scheme == 'euler':
            def step(x, dt, dw):
                return x + problem.a(x)*dt + np.dot(problem.b(x),dw)
        elif self.scheme in ['milstein', 'commutative_milstein']:
            def step(x, dt, dw, i):
                return x + problem.a(x)*dt + np.dot(problem.b(x),dw) + np.tensordot( problem.bp(x) , i )
        else:
            raise KeyError('Unknown scheme name')

        #jax.jit(step) # ####### TODO #########

        return step

    def solve(self, problem: VectorSDEProblem, wiener: VectorWiener = None):
        '''
        Produce one realisation of the process specified by ``problem``.

        Parameters
        ----------
        problem : VectorSDEProblem
             Vector stochastic differential equation together with bounary conditions.
        wiener : VectorWiener, optional
             Underlying Wiener processes supplying noise to the equation. Usefull when comparing several solvers or several equations on the same noise.

        Returns
        -------
        dict
            Dictionary containing 2 entries. Under key ``time_values`` a np.array of timestamps on which process was evaluated.
            Under key ``solution_values`` a np.array of stochastic process values at corresponding time instances.

        Example
        -------
        >>> import numpy as np
        >>> problem = pychastic.sde_problem.VectorSDEProblem(lambda x: np.array([1,1]), lambda x: np.array([[1,0.5],[0.5,1]]), 2, 2, np.array([1.5,0.5]), 1)
        >>> solver = pychastic.sde_solver.VectorSDESolver()
        >>> solver.solve(problem)
        {'time_values': array([0.,0.01,...]), 'solution_values' : array([[1.5, 0.5], [1.76, 0.71], [1.93, 0.91], ...])} #some values random

        '''
        if self.scheme == 'euler':
            wiener = wiener or VectorWiener(problem.noiseterms)
        elif self.scheme == 'commutative_milstein':
            wiener = wiener or VectorWiener(problem.noiseterms)
        elif self.scheme == 'milstein':
            wiener = wiener or VectorWienerWithI(problem.noiseterms)
        else:
            raise KeyError('Unknown scheme name: '+str(self.scheme))

        step = self.get_step_function(problem)

        dt = self.dt
        t = 0.0
        x = problem.x0

        # initialize "trajectories"
        time_values = [0.0]
        solution_values = [x]

        while True:
            w_prev = wiener.get_w(t)
            w_next = wiener.get_w(t+dt) # Careful to sample in correct order!
            dw = w_next - w_prev
            t += dt

            if self.scheme == 'euler':
                x = step(x, dt, dw)
            elif self.scheme == 'commutative_milstein':
                comm_noise = wiener.get_commuting_noise(t,t+dt)
                x = step(x, dt, dw, comm_noise)
            elif self.scheme == 'milstein':
                i = wiener.get_I_matrix(t, t+dt)
                x = step(x, dt, dw, i)

            solution_values.append(x)
            time_values.append(t)

            if t >= problem.tmax:
                break

        return dict(
            time_values=np.array(time_values),
            solution_values=np.array(solution_values)
        )
