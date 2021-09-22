import numpy as np

class normal:
  def __init__(self, seed=None):
    self.rng = np.random.default_rng(seed)
    self.n = 1
    self.buffer = np.array([])

  #@profile
  def _sample_more(self):
    new_things = self.rng.normal(size=self.n)
    self.buffer = np.concatenate([self.buffer, new_things])
    self.n *= 2
  
  #@profile
  def get_sample(self, size=1, n=None):
    '''
    Return np.array of Gaussians of specified shape

    Parameters
    ----------
    size : int or tuple of integers

    '''
    n = np.prod(size) if n is None else n
    while n > len(self.buffer):
      self._sample_more()
    x = self.buffer[:n]
    self.buffer = self.buffer[n:]
    return x.reshape(size)


  def get_number_of_samples(self, n):
    '''
    Return np.array of Gaussians of specified length

    Parameters
    ----------
    n : int or tuple of integers

    '''
    while n > len(self.buffer):
      self._sample_more()
    x = self.buffer[:n]
    self.buffer = self.buffer[n:]
    # return x.reshape(n) # slow :<
    return x

  #@profile
  def get_single_sample(self):
    '''
    Return a float
    '''
    if len(self.buffer) == 0:
      self._sample_more()
    x = self.buffer[0]
    self.buffer = self.buffer[1:]
    # return x.reshape(n) # slow :<
    return x

  def __iter__(self):
    return self
  
  def __next__(self):
      return self.get_single_sample()
