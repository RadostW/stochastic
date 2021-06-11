import unittest
from sortedcontainers.sorteddict import SortedDict

from pychastic.wiener import Wiener
from pychastic.wiener import WienerWithZ
from pychastic.wiener import VectorWienerWithI

import random
import numpy as np


random.seed(0)
np.random.seed(0)

class TestWiener(unittest.TestCase):
    def test_sampling_persistent(self):
        w = Wiener()

        w_first_try = w.get_w(float(7))

        T = 1000
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        w_second_try = w.get_w(float(7))

        self.assertEqual( w_first_try , w_second_try, 'Wiener value changed between samplings')

    def test_increment_variance(self):
        w = Wiener()
        T = 100
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))

        var = np.var(np.array(dw_list))

        self.assertAlmostEqual( var , 1 , delta=5.0/np.sqrt(T), 
               msg = f'Variance of Wiener increments incorrect: {var}')

        # add points in order
        dt = 0.01
        np.arange(0, T, dt)
        values = [w.get_w(t) for t in points]

        dw_list = []
        for t in np.arange(0, T, dt):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))

        var = np.var(dw_list)

        self.assertAlmostEqual( var , dt , delta=5.0*np.sqrt(len(np.arange(0, T, dt))), 
               msg = f'Variance of Wiener increments incorrect: {var}')

    def test_autocovariance(self):
        w = Wiener()
        T = 100
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))
        dw_list = np.array(dw_list)

        #Compute covariance of two random samples
        cov = np.correlate(dw_list[1:],dw_list[:-1])[0] / (len(dw_list)-2)

        self.assertAlmostEqual( cov , 0 , delta=5.0/np.sqrt(T), 
               msg = f'Autocovariance of Wiener increments incorrect: {cov}')


class TestWienerWithZ(unittest.TestCase):
    def test_sampling_persistent(self):
        w = WienerWithZ()

        w_first_try = w.get_w(float(7))

        T = 1000
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        w_second_try = w.get_w(float(7))

        self.assertEqual( w_first_try , w_second_try, 'Wiener value changed between samplings')


    def test_sampling_z_persistent(self):
        w = WienerWithZ()

        z_first_try = w.get_z(float(7),float(10))

        T = 1000
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        z_second_try = w.get_z(float(7),float(10))

        assert np.isclose(z_first_try, z_second_try, atol=1e-6), 'Z integral value changed between samplings'

    def test_ensure_sample_point_properly_distributes_z(self):
        w = WienerWithZ()
        t1, t2, t3 = 1, 2, 3
        w.ensure_sample_point(t1)
        w.ensure_sample_point(t3)
        z13 = w.sample_points[t3]['zToPrevPoint']
        w.ensure_sample_point(t2)
        z12 = w.sample_points[t2]['zToPrevPoint']
        z23 = w.sample_points[t3]['zToPrevPoint']
        w1 = w.sample_points[t1]['w']
        w2 = w.sample_points[t2]['w']
        w3 = w.sample_points[t3]['w']
        new_z13 = z12 + z23 + (t3-t2)*(w2-w1)
        assert np.isclose(z13, new_z13)

    def test_z_sums_properly(self):
        np.random.seed(0)
        t1, t2, t3, t4 = 1, 2, 3, 4
        w1, w2, w3, w4 = np.random.normal(size=4)
        z01, z12, z23, z34 = np.random.normal(size=4)
        w = WienerWithZ()
        w.sample_points = SortedDict({
            0:  {'w': 0, 'zToPrevPoint': 0},
            t1: {'w': w1, 'zToPrevPoint': z01},
            t2: {'w': w2, 'zToPrevPoint': z12},
            t3: {'w': w3, 'zToPrevPoint': z23},
            t4: {'w': w4, 'zToPrevPoint': z34},
        })
        true_z14 = z12 + z23 + (t3-t2)*(w2-w1) + z34 + (t4-t3)*(w3-w1)
        z14 = w.get_z(t1, t4)
        assert np.isclose(true_z14, z14)


    
    def test_z_variance_and_covariance(self):
        w = WienerWithZ()
        T = 10000
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        dz_list = []
        for t in range(T):
            dz_list.append( w.get_z(float(t),float(t+1)) )

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))

        var = np.var(np.array(dz_list))
        covar = np.correlate(np.array(dz_list),np.array(dw_list))[0] / (len(dw_list)-1)

        self.assertAlmostEqual( var , 1.0/3.0 , delta=5.0/np.sqrt(T), 
               msg = f'Variance of Z increments incorrect: {var}')

        self.assertAlmostEqual( covar , 1.0/2.0 , delta=5.0/np.sqrt(T), 
               msg = f'Covariance of Z increments incorrect')


    def test_increment_variance(self):
        w = WienerWithZ()
        T = 100
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))

        var = np.var(np.array(dw_list))

        self.assertAlmostEqual( var , 1 , delta=5.0/np.sqrt(T), 
               msg = f'Variance of Wiener increments incorrect: {var}')

        # add points in order
        dt = 0.01
        np.arange(0, T, dt)
        values = [w.get_w(t) for t in points]

        dw_list = []
        for t in np.arange(0, T, dt):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))

        var = np.var(dw_list)

        self.assertAlmostEqual( var , dt , delta=5.0*np.sqrt(len(np.arange(0, T, dt))), 
               msg = f'Variance of Wiener increments incorrect: {var}')

    def test_autocovariance(self):
        w = WienerWithZ()
        T = 100
        points = list(range(T))

        # add points to lookup in random order
        random.shuffle(points)
        for t in points:
            w.get_w(float(t))

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))
        dw_list = np.array(dw_list)

        #Compute covariance of two random samples
        cov = np.correlate(dw_list[1:],dw_list[:-1])[0] / (len(dw_list)-2)

        self.assertAlmostEqual( cov , 0 , delta=5.0/np.sqrt(T), 
               msg = f'Autocovariance of Wiener increments incorrect: {cov}')


    def test_vector_double_integrals(self):
        '''
        E(I_12) = 0, E(I_12 ^2) = h^2 / 2
        E(I_12 I_13) = E(I_1 I_12) = E(I_12 I_21) = 0
        I_12 + I_21 = I_1 * I_2
        '''
        w = VectorWiener(noiseterms=2)
        T = 100
        points = list(range(T))
        for t in points:
            w.get_w(float(t))

        dw_list = []
        for t in range(T):
            dw_list.append( w.get_w(float(t+1)) - w.get_w(float(t)))
        dw_list = np.array(dw_list)

class TestVectorWienerWithI(unittest.TestCase):
    def testIMoments(self):
        w = VectorWienerWithI(noiseterms=2)
        n = 10000
        dt = 0.1
        sigma = dt/np.sqrt(2)
        points = np.arange(n+1)*dt
        for t in points:
            w.get_w(t)
        
        data = np.stack([w.get_I_matrix(points[i], points[i+1]) for i in range(len(points)-1)])
        assert np.isclose(data.mean(axis=0),[
            [0, 0],
            [0, 0]
        ], atol=5*sigma/np.sqrt(n)).all()

        m11 = dt**2/2
        m12 = dt**2/2
        assert np.isclose((data**2).mean(axis=0),[
            [m11, m12],
            [m12, m11]
        ], atol=5*sigma/np.sqrt(n)).all()

        
        m11 = 15/4*dt**4
        m12 = 7/4*dt**4
        assert np.isclose((data**4).mean(axis=0),[
            [m11, m12],
            [m12, m11]
        ], atol=5*68*dt**4).all()


if __name__ == '__main__':
    #np.random.seed(0)
    unittest.main()
