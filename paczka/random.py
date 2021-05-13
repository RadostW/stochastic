import numpy as np

def normal(seed=None):
  if seed:
    np.random.seed(seed)
  size = 1
  samples = []
  while True:
    if len(samples) == 0:
      samples = np.random.normal(size=size)
      size *= 2
    yield samples[-1]
    samples = samples[:-1]
