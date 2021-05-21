import numpy as np

class normal:
  def __init__(self, seed=None):
    self.rng = np.random.default_rng(seed)
    self.n = 1
    self.buffer = np.array([])

  def _sample_more(self):
    self.buffer = np.concatenate([self.buffer, self.rng.normal(size=self.n)])
    self.n *= 2
    
  def get_sample(self, size=1):
    '''
    Parameters
    ----------
    size : int or tuple of integers

    '''
    n = np.prod(size)
    while n > len(self.buffer):
      self._sample_more()
    x = self.buffer[:n]
    self.buffer = self.buffer[n:]
    return x.reshape(size)

  def __iter__(self):
    return self
  
  def __next__(self):
      return self.get_sample(1).item()
