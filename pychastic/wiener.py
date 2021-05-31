from numpy.core.fromnumeric import transpose
import sortedcontainers
from pychastic.cached_gaussian import normal
import numpy as np
import jax.numpy as jnp
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
        '''
        Get value of Wiener probess at specified timestamp.

        Parameters
        ----------
        t: float
            Time at which the process should be sampled. Has to be non-negative.

        Returns
        -------
        float
            Value of Wiener process at time ``t``.

        Example
        -------
        >>> wiener = Wiener()
        >>> dW = wiener.get_w(1.0) - wiener.get_w(0.0)
        >>> dW
        0.321 #random value from N(0,1)


        '''
        if not t >= 0:
            raise ValueError('Illegal (negative?) timestamp')

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

    def get_w(self,t):
        '''
        Get value of Wiener probess at specified timestamp.

        Parameters
        ----------
        t: float
            Time at which the process should be sampled. Has to be non-negative.

        Returns
        -------
        float
            Value of Wiener process at time ``t``.

        Example
        -------
        >>> wiener = WienerWithZ()
        >>> dW = wiener.get_w(1.0) - wiener.get_w(0.0)
        >>> dW
        0.321 #random value from N(0,1)


        '''
        self.ensure_sample_point(t)
        return self.sample_points[t]['w']

    def get_z(self,t1,t2):
        '''
        Get value of first nontrivial, primitive stochastic integral I(1,0),
        (Kloden-Platen 10.4.2)

        .. math ::  Z_{t_1}^{t_2} = \int_{t_1}^{t_2} \int_{t_1}^{s_2} dW_{s_1} ds_2

        Parameters
        ----------
        t: float
            Time at which the process should be sampled. Has to be non-negative.

        Returns
        -------
        float
            Value of Wiener process at time ``t``.

        Example
        -------
        >>> wiener = WienerWithZ()
        >>> dZ = wiener.get_z(0.0,0.1)
        >>> dZ
        0.321 #random value from N(0,1)


        '''
        if t1 >= t2:
            raise ValueError

        self.ensure_sample_point(t1)
        self.ensure_sample_point(t2)

        Z = 0
        w1 = self.sample_points[t1]['w']
        it_lower = self.sample_points.irange(t1, t2)
        it_upper = self.sample_points.irange(t1, t2)
        next(it_upper)
        for t_upper in it_upper:
            t_lower = next(it_lower)
            Z += self.sample_points[t_upper]['zToPrevPoint']
            dt = t_upper-t_lower
            dw = self.sample_points[t_lower]['w'] - w1
            Z += dw*dt

        return Z


    def ensure_sample_point(self,t):
        '''
        Ensures ``t`` is in the dictionary of sampled time instances. If not there yet samples new point 
        either to the right of all existing points or inbetween existing points.
        '''
        if t in self.sample_points.keys():
            return

        if t < 0:
            raise ValueError

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

            wt1t3 = next_w - prev_w
            zt1t3 = self.sample_points.peekitem(next_i)[1]['zToPrevPoint']

            (t1,t2,t3) = (prev_t,t,next_t)
            i1 = t2-t1
            i2 = t3-t2
            I = t3-t1

            varwt1t2 =  i1*i2*(i1*i1-i1*i2+i2*i2)/(I*I*I);
            cov = i1*i1*i2*i2*(i2-i1)/(2*I*I*I);
            varzt1t2 = i1*i1*i1*i2*i2*i2/(3*I*I*I);

            (wt1t2, zt1t2) = self._DrawCovaried(varwt1t2,cov,varzt1t2)

            #Add conditional mean
            wt1t2 += wt1t3*i1*(i1-2*i2)/(I*I)  + zt1t3*6*i1*i2/(I*I*I)
            zt1t2 += wt1t3*(-1)*i1*i1*i2/(I*I) + zt1t3*i1*i1*(i1+3*i2)/(I*I*I)

            wt2 = prev_w + wt1t2

            #Break Z integration interval into two segments
            self.sample_points[next_t]['zToPrevPoint'] = (
                      self.sample_points.peekitem(next_i)[1]['zToPrevPoint']
                      - zt1t2
                      - (next_t-t)*(wt2-prev_w))
            self.sample_points[t] = {'w' : wt2, 'zToPrevPoint' : zt1t2}

    def _DrawCovaried(self,xx,xy,yy):
        '''
        Draws x,y from N(0,{{xx,xy},{xy,yy}}) distribution
        '''
        (x,y) = self._DrawCorrelated(xy/math.sqrt(xx*yy))
        return (math.sqrt(xx)*x,math.sqrt(yy)*y)

    def _DrawCorrelated(self,cor):
        '''
        Draws correalted normal samples with correlation ``cor``
        '''
        z1 = next(self.normal_generator)
        z2 = next(self.normal_generator)
        return (math.sqrt(1-cor*cor)*z1 + cor*z2,z2)


class VectorWiener:
    '''
    Class for sampling, and memorization of vector valued Wiener process.


    Parameters
    ----------
    noiseterms : int
        Dimensionality of the vector process (i.e. number of independent Wiener processes).

    Example
    -------
    >>> vw = pychastic.wiener.VectorWiener(2)
    >>> vw.get_w(1)
    array([0.21,-0.31]) # random, independent from N(0,1)

    '''
    def __init__(self,noiseterms : int):
        self.sample_points = sortedcontainers.SortedDict()
        self.noiseterms = noiseterms
        self.sample_points[0.] = {
            'w': np.array([0.0 for x in range(0,noiseterms)]),
        }
        self.normal_generator = normal()

    def get_w(self, t):
        '''
        Get value of Wiener probess at specified timestamp.

        Parameters
        ----------
        t: float
            Time at which the process should be sampled. Has to be non-negative.

        Returns
        -------
        np.array
            Value of Wiener processes at time ``t``.

        Example
        -------
        >>> vw = VectorWiener(2)
        >>> dW = vw.get_w(1.0) - vw.get_w(0.0)
        >>> dW
        array([0.321,-0.123]) #random, each from N(0,1)

        '''
        if t < 0:
            raise ValueError('Negative timestamp')

        if t in self.sample_points:
            return self.sample_points[t]['w']

        t_max = self.sample_points.keys()[-1]
        if t > t_max:
            nvec = np.array([next(self.normal_generator) for x in range(0,self.noiseterms)])
            self.sample_points[t] = {'w': np.array(self.sample_points[t_max]['w'] + np.sqrt(t-t_max)*nvec)}
        else:
            #print(f'Called with t {t}, current t_max is {t_max}')
            #print(self.sample_points)
            raise NotImplementedError

        return self.sample_points[t]['w']
    def get_commuting_noise(self, t1, t2):
        '''
        Get value of commutative noise matrix (compare Kloden-Platen (10.3.15))

        Define :math:`I_{jk}` as

        .. math :: I_{jk}(t_1,t_2) = \int_{t_1}^{t_2} \int_{t_1}^{s_1} dW_j(s_2) dW_k(s_1)

        Then for :math:`j \\neq k`

        .. math :: I_{jk} + I_{kj} = \Delta W_j \Delta W_k

        Parameters
        ----------
        t1 : float
            Lower bound of double stochastic integrals
        t2 : float
            Upper bound of double stochastic integrals

        Returns
        -------
        np.array
            Symmetric square matrix `noiseterms` by `noiseterms` containing :math:`I_{jk}` approximants as components.

        '''

        if t1 < 0 or t2 < 0:
            raise ValueError

        if t1 > t2:
            raise ValueError

        if (t1 in self.sample_points) and (t2 in self.sample_points):
            dW = self.sample_points[t2]['w'] - self.sample_points[t1]['w']
            dV = self.sample_points[t2]['w'] - self.sample_points[t1]['w']

            prod = np.outer(dW,dV)
            #halfdiag = np.oneslike(prod) - 0.5*np.identity(self.noiseterms)

            #return prod*halfdiag - (t2-t1)*np.identity(self.noiseterms)
            return 0.5*prod


        t_max = self.sample_points.keys()[-1]
        if t1 > t_max:
            nvec = np.array([next(self.normal_generator) for x in range(0,self.noiseterms)])
            self.sample_points[t1] = {'w': self.sample_points[t_max]['w'] + np.sqrt(t1-t_max)*nvec}
        elif t1 not in self.sample_points:
            raise NotImplementedError

        if t2 > t_max:
            nvec = np.array([next(self.normal_generator) for x in range(0,self.noiseterms)])
            self.sample_points[t2] = {'w': self.sample_points[t_max]['w'] + np.sqrt(t2-t_max)*nvec}
        elif t2 not in self.sample_points:
            raise NotImplementedError

        dW = self.sample_points[t2]['w'] - self.sample_points[t1]['w']
        dV = self.sample_points[t2]['w'] - self.sample_points[t1]['w']

        prod = np.outer(dW,dV)
        #halfdiag = np.oneslike(prod) - 0.5*np.identity(self.noiseterms)

        #return prod*halfdiag - (t2-t1)*np.identity(self.noiseterms)
        return 0.5*prod

    def get_commuting_noise_component(self, t1, t2, j, k):
        '''
        Get value of commutative noise component (compare Kloden-Platen (10.3.15)).

        Define :math:`I_{jk}` as

        .. math :: I_{jk}(t_1,t_2) = \int_{t_1}^{t_2} \int_{t_1}^{s_1} dW_j(s_2) dW_k(s_1)

        Then for :math:`j \\neq k`

        .. math :: I_{jk} + I_{kj} = \Delta W_j \Delta W_k

        Parameters
        ----------
        t1 : float
            Lower bound of double stochastic integrals
        t2 : float
            Upper bound of double stochastic integrals
        j : int
            Index of the first of Wiener processes
        k : int
            Index of the second of Wiener processes

        Returns
        -------
        float
            Value of stochastic integral with specified time bounds.        

        '''

        if t1 < 0 or t2 < 0:
            raise ValueError

        if t1 > t2:
            raise ValueError

        if (t1 in self.sample_points) and (t2 in self.sample_points):
            dW = self.sample_points[t2]['w'][j] - self.sample_points[t1]['w'][j]
            dV = self.sample_points[t2]['w'][k] - self.sample_points[t1]['w'][k]
            if j != k:
                return 0.5*dW*dV
            else:
                return 0.5*(dW*dW - (t2-t1))


        t_max = self.sample_points.keys()[-1]
        if t1 > t_max:
            nvec = np.array([next(self.normal_generator) for x in range(0,self.noiseterms)])
            self.sample_points[t1] = {'w': self.sample_points[t_max]['w'] + np.sqrt(t1-t_max)*nvec}
        elif t1 not in self.sample_points:
            raise NotImplementedError

        if t2 > t_max:
            nvec = np.array([next(self.normal_generator) for x in range(0,self.noiseterms)])
            self.sample_points[t2] = {'w': self.sample_points[t_max]['w'] + np.sqrt(t2-t_max)*nvec}
        elif t2 not in self.sample_points:
            raise NotImplementedError

        dW = self.sample_points[t2]['w'][j] - self.sample_points[t1]['w'][j]
        dV = self.sample_points[t2]['w'][k] - self.sample_points[t1]['w'][k]

        if j != k:
            return dW*dV
        else:
            return 0.5*(dW*dW - (t2-t1))


class VectorWienerWithI:
    '''
    Class for sampling, and memorization of vector valued Wiener process including first
    nontrivial vector stochastic integral :math:`I_{jk}`.


    Parameters
    ----------
    noiseterms : int
        Dimensionality of the vector process (i.e. number of independent Wiener processes).
    p : int, default: 10
        Number of terms in fourier expansion of the stochastic integral. 
        The higher the number, the more precise approximation becomes.

    Example
    -------
    >>> vw = pychastic.wiener.VectorWienerWithJ(2)
    >>> vw.get_w(1)
    array([0.21,-0.31]) # random, independent from N(0,1)

    '''
    def __init__(self,noiseterms : int, p = 10):
        self.sample_points = sortedcontainers.SortedDict()
        self.noiseterms = noiseterms
        self.sample_points[0.] = {
            'w': np.zeros(noiseterms),
            'IToPrevPoint' : np.zeros((noiseterms,noiseterms))
        }
        self.normal_generator = normal()
        self.p = p

    def get_w(self, t):
        '''
        Get value of Wiener probess at specified timestamp.

        Parameters
        ----------
        t: float
            Time at which the process should be sampled. Has to be non-negative.

        Returns
        -------
        np.array
            Value of Wiener processes at time ``t``.

        Example
        -------
        >>> vw = VectorWienerWithI(2)
        >>> dW = vw.get_w(1.0) - vw.get_w(0.0)
        >>> dW
        array([0.321,-0.123]) #random, each from N(0,1)

        '''
        if t < 0:
            raise ValueError('Negative timestamp')

        if t in self.sample_points:
            return self.sample_points[t]['w']

        t_max = self.sample_points.keys()[-1]
        if t > t_max:
            self.ensure_sample_point(t)
        else:
            next_i = self.sample_points.bisect_left(t)
            next_t = self.sample_points.peekitem(next_i)[0]
            prev_t = self.sample_points.peekitem(next_i-1)[0]
            if ( next_t - t < 2*jnp.finfo(type(t)).eps ):
                return self.sample_points[next_t]['w']
            elif ( t - prev_t < 2*jnp.finfo(type(t)).eps ):
                return self.sample_points[prev_t]['w']
            else:
                raise NotImplementedError('Conditional subsampling not implemented.')
                # #### TODO #### Implement subsampling

        return self.sample_points[t]['w']

    def get_I_matrix(self, t1, t2):
        '''
        Get value of double integrals :math:`I_{jk}` (compare Kloden-Platen (10.3.5))

        Define :math:`I_{jk}` as

        .. math :: I_{jk}(t_1,t_2) = \int_{t_1}^{t_2} \int_{t_1}^{s_1} dW_j(s_2) dW_k(s_1)

        Parameters
        ----------
        t1 : float
            Lower bound of double stochastic integrals
        t2 : float
            Upper bound of double stochastic integrals

        Returns
        -------
        np.array
            Symmetric square matrix `noiseterms` by `noiseterms` containing :math:`I_{jk}` approximants as components.

        '''

        if not (t1 >= 0 and t2 >= 0):
            raise ValueError('Illegal timestamps for sampling. (Negative?)')

        if t1 > t2:
            raise ValueError('Wrong order of integration limits.')

        t_max = self.sample_points.keys()[-1]
        if t1 > t_max:
            self.ensure_sample_point(t1)
        elif t1 not in self.sample_points:
            next_i = self.sample_points.bisect_left(t1)
            next_t = self.sample_points.peekitem(next_i)[0]
            prev_t = self.sample_points.peekitem(next_i-1)[0]
            if ( next_t - t1 < 2*jnp.finfo(type(t1)).eps ):
                t1 = next_t
            elif ( t1 - prev_t < 2*jnp.finfo(type(t1)).eps ):
                t1 = prev_t
            else:
                raise NotImplementedError('Conditional subsampling not implemented.')

        if t2 > t_max:
            self.ensure_sample_point(t2)
        elif t2 not in self.sample_points:
            next_i = self.sample_points.bisect_left(t2)
            next_t = self.sample_points.peekitem(next_i)[0]
            prev_t = self.sample_points.peekitem(next_i-1)[0]
            if ( next_t - t2 < 2*jnp.finfo(type(t2)).eps ):
                t2 = next_t
            elif ( t2 - prev_t < 2*jnp.finfo(type(t2)).eps ):
                t2 = prev_t
            else:
                raise NotImplementedError('Conditional subsampling not implemented.')

        # Recall: I(1->3)  = I(1->2) + I(2->3) + dWa(2->3) dWb(1->2)

        I = 0
        w1 = self.sample_points[t1]['w']
        it_lower = self.sample_points.irange(t1, t2)
        it_upper = self.sample_points.irange(t1, t2)
        next(it_upper)
        for t_upper in it_upper:
            t_lower = next(it_lower)
            I += self.sample_points[t_upper]['IToPrevPoint']            
            dw = self.sample_points[t_upper]['w'] - self.sample_points[t_lower]['w']
            dw_from_start = self.sample_points[t_lower]['w'] - w1
            I += dw*dw_from_start

        dw = self.sample_points[t2]['w'] - self.sample_points[t1]['w']
        np.fill_diagonal(I, 0.5*(dw**2 - (t2-t1))) # Diagonal entries work differently

        return I

    def ensure_sample_point(self,t):
        '''
        Checks if sample point exists, adds new sample if necessary

        Parameters
        ----------
        t : float
            Timestamp to potentially add to sample history

        '''

        if not (t>=0):
            raise ValueError('Illegal timestamps for sampling. (Negative?)')
        if t in self.sample_points:
            return

        t_max = self.sample_points.keys()[-1]

        # ##### TODO ###### change implemntation to use cached normals
        # Compare Kloden-Platen (10.3.7), dimension = d, noiseterms = m
        Delta = t - t_max

        dW = np.sqrt(Delta)*np.array([next(self.normal_generator) for x in range(0, self.noiseterms)])
        xi = (1.0 / np.sqrt(Delta) * dW).reshape(1, -1)
        mu = np.random.normal(size=self.noiseterms).reshape(1, -1)
        eta = np.random.normal(size=(self.p, self.noiseterms))
        zeta = np.random.normal(size=(self.p, self.noiseterms))
        rec = 1/np.arange(1, self.p+1) # 1/r vector
        rho = 1/12 - (rec**2).sum()/(2*np.pi**2)

        a = math.sqrt(2)*xi+eta

        Imat = (
            Delta*(xi*xi.T/2 + np.sqrt(rho)*(mu*xi.T - xi*mu.T))
            + Delta/(2*np.pi)*  ((np.expand_dims(a, 1)*np.expand_dims(zeta, 2) - np.expand_dims(a, 2)*np.expand_dims(zeta, 1)).T*rec).sum(axis=-1)
        )
        
        np.fill_diagonal(Imat, 0.5*(dW**2 - Delta)) # Diagonal entries work differently

        self.sample_points[t] = {'w': self.sample_points[t_max]['w'] + dW, 'IToPrevPoint' : Imat}



