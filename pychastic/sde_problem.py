import jax
import numpy as np

class SDEProblem:
  '''
  Stores a stochastic differential equation of the form:
  :math:`dX = a(X) dt + b(X) dW`, where ``a`` and ``b`` are
  (possibly positionally dependent) drift and noise coefficients.

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

  Example
  -------

  >>> problem = pychastic.sde_problem.SDEProblem(lambda x: 1.0,lambda x: -1.0,0.0,0.1)

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

class VectorSDEProblem:
  '''
  Stores a vector stochastic differential equation of the form:
  :math:`d\\mathbf{x} = \\mathbf{a(x)} dt + \\mathbf{B(x)} d\\mathbf{w}`, where
  ``a`` and ``B`` are vector of drifts and matrix of noise coefficients.

  ``a`` vector and ``B`` matrix should be formatted such that

  .. prompt:: python >>> auto

    >>> x = x + a(x)*dt + np.dot(B(x),dw)

  is a valid code fragment (note ``B`` has as many columns as there are noise terms).

  Parameters
  ----------
  a : callable
    Function describing drift term of the equation, should return np.array of length ``dimension``.
  b : callable
      Function describing noise term of the equation, should return np.array of size ``(dimension,noiseterms)``.
  dimension : int
      Dimension of the space in which :math:`d\\mathbf{x}` sits.
  noiseterms : int
      Dimension of the space in which :math:`d\\mathbf{w}` sits.
  x0 : np.array
      Initial value of the stochastic process.
  tmax : float
      Time at which integration should stop.

  Example
  -------
  >>> import numpy as np
  >>> problem = pychastic.sde_problem.VectorSDEProblem(lambda x: np.array([1,1]), lambda x: np.array([[1,0.5],[0.5,1]]), 2, 2, np.array([1.5,0.5]), 1)

  '''
  def __init__(self, a, b, dimension, noiseterms, x0, tmax):
    self.a = a
    self.b = b
    self.dimension = dimension
    self.noiseterms = noiseterms
    self.x0 = x0
    self.tmax = tmax

    tmpa = a(x0)
    tmpb = b(x0)

    assert isinstance(tmpa, np.ndarray) , 'Drift term should return np.array'
    assert np.issubdtype(tmpa.dtype, np.floating) , f'Drift term should be array of floats, not {tmpa.dtype}.'
    assert tmpa.shape == (dimension,) , f'Drift term should be array of shape (dimension,) == {(self.dimension,)}, not {tmpa.shape}'

    assert isinstance(tmpb, np.ndarray) , 'Noise term should return np.array'
    assert np.issubdtype(tmpb.dtype, np.floating) , f'Noise term should be array of floats, not {tmpb.dtype}.'
    assert tmpb.shape == (dimension,noiseterms) , f'Drift term should be array of shape (dimension, noiseterms) == {(self.dimension,self.noiseterms)}, not {tmpb.shape}'
    

    self.ap = jax.jacfwd(a)
    self.bp = jax.jacfwd(b)

