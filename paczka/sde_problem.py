import jax

class SDEProblem:
  '''
  Stores an SD equation, does some derivative precomputing on init.
  exact_solution: (x0, time_values, wiener_values) -> solution_values
  '''
  def __init__(self, a, b, x0, tmax, exact_solution=None):
    self.a = a
    self.b = b
    self.tmax = tmax
    self.x0 = x0
    self.exact_solution = exact_solution

    self.ap = jax.grad(a)
    self.bp = jax.grad(b)
    self.app = jax.grad(self.ap)
    self.bpp = jax.grad(self.bp)
