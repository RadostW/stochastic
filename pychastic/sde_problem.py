import jax

class SDEProblem:
  '''

  Stores a stochastic differential equation of the form:
  dX = a(X) dt + b(x) dW

  ...

  Parameters
  ----------
  a : callable
      Function describing drift term of the equation
  b : callable
      Function describing noise term of the equation
  x0 : float
      Initial value of the stochastic process
  tmax : float
      Time at which integration should stop
  exact_solution : callable, optional
      Exact solution of the SDE for comparing accuracies. If function ``f`` 
      is passed as exact solution it should satisty ``[X(t1),X(t2),...] = f(x0, [t1,t2,t3], [W(t1),W(t2),...])`` 
      where W is value of the underlying Wiener process forcing the equation

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
