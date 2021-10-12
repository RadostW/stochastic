import jax
import jax.numpy as jnp


class SDEProblem:
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
  def __init__(self, a, b, x0, tmax, exact_solution=None):
    
    if not tmax > 0:
      raise ValueError(f'tmax has to bo posiitve, not {tmax}')
    self.tmax = tmax
    
    # dimension & shape validation
        
    x0 = jnp.array(x0, dtype=jax.numpy.float32)

    if x0.ndim == a(x0).ndim == b(x0).ndim == 0:
      self.x0 = x0.reshape(1)
      # scalar case
      if a(x0).ndim != 1:
        new_a = lambda x: a(x).reshape(1)
        # TODO warn about reshaping
      if b(x0).ndim != 2:
        new_b = lambda x: b(x).reshape(1, 1)
        # TODO warn about reshaping
      
      self.dimension = self.noise_terms = 1
      self.a = new_a
      self.b = new_b

    elif x0.ndim == a(x0).ndim == 1 and b(x0).ndim == 2:
      # vector case
      if not x0.shape[0] == a(x0).shape[0] == b(x0).shape[0]:
        raise ValueError(f'Incosistent shapes: {x0.shape}, {a(x0).shape}, {b(x0).shape}')   

      self.x0 = x0
      self.a = a
      self.b = b
      self.dimension, self.noise_terms = b(x0).shape   

    else:
      raise ValueError(f'Inconsistent dimensions: {x0.shape}, {a(x0).shape}, {b(x0).shape}')
      
    
    # dtype validation

    for val, key in [(x0, 'initial conditon'), (a(x0), 'drift term'), (b(x0), 'noise term')]:
      if not isinstance(val, jnp.ndarray):
        raise ValueError(f"{key} should return jnp.array, not {type(val)}")
      if not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError(f"{key} dtype should be float, not {val.dtype}.")

    # exact solution validation
    # TODO
    self.exact_solution = exact_solution
