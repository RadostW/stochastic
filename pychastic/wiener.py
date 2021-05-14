import sortedcontainers
from pychastic.cached_gaussian import normal
import numpy as np
import math

class Wiener:
    '''
    Class for sampling, and memorization of Wiener process.
    '''
    def __init__(self):
        self.sample_points = sortedcontainers.SortedDict()
        self.sample_points[0.] = {
            'w': 0.0,
            #'zToPrevPoint': 0.0
        }
        self.normal_generator = normal()

    def get_w(self, t):
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


class WienerWithZ:
    '''
    Class for sampling, and memorization of Wiener process and first nontrivial Stochastic integral.
    '''
    def __init__(self):
        self.sample_points = sortedcontainers.SortedDict()
        self.sample_points[0.] = {
            'w': 0.0,
            'zToPrevPoint': 0.0
        }
        self.normal_generator = normal()

    def ensure_sample_point(self,t):
        '''
        Ensures ``t`` is in the dictionary of sampled time instances. If not there yet samples new point 
        either to the right of all existing points or inbetween existing points.
        '''
        if t < 0:
            return ValueError

        t_max = self.sample_points.keys()[-1]
        if t > t_max:
            #Kloden-Platen 10.4.3
            # (4.3)       dW = U1 sqrt(dt), dZ = 0.5 dt^(3/2) (U1 + 1/sqrt(3) U2)
            tmpU1 = next(self.normal_generator)
            tmpU2 = next(self.normal_generator)

            tmpdt = t - t_max
            tmpdW = tmpU1*math.sqrt(tmpdt)
            tmpdZ = 0.5*math.pow(tmpdt,3.0/2.0)*(tmpU1 + (1.0 / math.sqrt(3))*tmpU2 )

            self.sample_points[t] = {'w': self.sample_points[t_max]['w'] + tmpdW, 'zToPrevPoint': tmpdZ}
        else:
            #Somewhere inside sampled points
            next_i = self.sample_points.bisect_left(t)
            next_t = self.sample_points.peekitem(next_i)[0]
            prev_t = self.sample_points.peekitem(next_i-1)[0]
            next_w = self.sample_points.peekitem(next_i)[1]['w']
            prev_w = self.sample_points.peekitem(next_i-1)[1]['w']

            i1 = t2-t1
            i2 = t3-t2
            I = t3-t1

            varwt1t2 =  i1*i2*(i1*i1-i1*i2+i2*i2)/(I*I*I);
            cov = i1*i1*i2*i2*(i2-i1)/(2*I*I*I);
            varzt1t2 = i1*i1*i1*i2*i2*i2/(3*I*I*I);

            (wt1t2, zt1t2) = DrawCovaried(varwt1t2,cov,varzt1t2)

            #Add conditional mean
            wt1t2 += wt1t3*i1*(i1-2*i2)/(I*I)  + zt1t3*6*i1*i2/(I*I*I)
            zt1t2 += wt1t3*(-1)*i1*i1*i2/(I*I) + zt1t3*i1*i1*(i1+3*i2)/(I*I*I)

            wt2 = prew_w + wt1t2

            #Break Z integration interval into two segments
            self.sample_points.peekitem(next_i)['zToPrevPoint'] = (
                      self.sample_points.peekitem(next_i)['zToPrevPoint']
                      - zt1t2
                      - (next_t-t)*(wt2-prev_w))
            self.sample_points[t] = {'w' : wt2, 'zToPrevPoint' : zt1t2}

    def DrawCovaried(self,xx,xy,yy):
        '''
        Draws x,y from N(0,{{xx,xy},{xy,yy}}) distribution
        '''
        (x,y) = DrawCorrelated(xy/sqrt(xx*yy))
        return (math.sqrt(xx)*x,math.sqrt(yy)*y)

    def DrawCorrelated(self,cor):
        '''
        Draws correalted normal samples with correlation ``cor``
        '''
        z1 = next(self.normal_generator)
        z2 = next(self.normal_generator)
        return (math.sqrt(1-cor*cor)*z1 + cor*z2,z2)

