from paczka.wiener import Wiener
import random
import numpy as np


random.seed(0)
np.random.seed(0)

def test_wiener_increments():
    w = Wiener()
    T = 100
    points = list(range(T))
    random.shuffle(points)
    for t in points:
        w.get_w(float(t))

    dt = 0.01
    points = np.arange(0, T, dt)
    values = [w.get_w(t) for t in points]
    diff = np.diff(values)
    import matplotlib.pyplot as plt
    plt.plot(values)
    plt.savefig('w.png')
    assert np.isclose(np.var(diff), dt, rtol=0.05), np.var(diff)
    
test_wiener_increments()
