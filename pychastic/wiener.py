import sortedcontainers
from pychastic.cached_gaussian import normal
import numpy as np


class Wiener:
    def __init__(self, seed=0):
        self.sample_points = sortedcontainers.SortedDict()
        self.sample_points[0.] = {
            'w': 0.0,
            #'zToPrevPoint': 0.0
        }
        self.normal_generator = normal(seed=seed)
    
    def get_w(self, t):
        t = float(t)
        if t < 0:
            return ValueError
        
        if t in self.sample_points:
            return self.sample_points[t]['w']
        
        t_max = self.sample_points.keys()[-1]
        if t > t_max:
            self.sample_points[t] = {'w': self.sample_points[t_max]['w'] + np.sqrt(t-t_max)*next(self.normal_generator)}

        else:
            next_i = self.sample_points.bisect_left(t)
            next_t = self.sample_points.peekitem(next_i)[0]
            prev_t = self.sample_points.peekitem(next_i-1)[0]
            next_w = self.sample_points.peekitem(next_i)[1]['w']
            prev_w = self.sample_points.peekitem(next_i-1)[1]['w']
            w = prev_w + (t-prev_t)/(next_t-prev_t)*(next_w-prev_w) + next(self.normal_generator)*np.sqrt((next_t-t)*(t-prev_t)/(next_t-prev_t))
            assert np.isfinite(w)
            self.sample_points[t] = {'w': w}
        
        return self.sample_points[t]['w']


    def get_z(self, t1, t2):
        raise NotImplementedError
