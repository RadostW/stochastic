import unittest

from pychastic.wiener import Wiener
from pychastic.wiener import WienerWithZ

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

        self.assertEqual( z_first_try , z_second_try, 'Z integral value changed between samplings')

    def test_z_variance_and_covariance(self):
        w = WienerWithZ()
        T = 1000
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

        self.assertAlmostEqual( var , 1.0/2.0 , delta=5.0/np.sqrt(T), 
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

        

if __name__ == '__main__':
    unittest.main()
