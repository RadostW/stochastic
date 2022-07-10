# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

hist_small = np.genfromtxt('./sources/histogram_small_step.csv', delimiter=',')
hist_large = np.genfromtxt('./sources/histogram_large_step.csv', delimiter=',')

hist_teo = np.genfromtxt('./sources/hitting_time_teo.csv', delimiter=',')

plt.rcParams.update({
    "legend.frameon": False,
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 15,
    "figure.subplot.left":   0.125,
    "figure.subplot.right":  0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top":    0.95,
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.2,
})

n = len(hist_small)
xs = np.linspace(0,1.7,n)
dx = xs[1]-xs[0]
plt.bar(xs-0.25*dx,hist_small,width=0.5*dx, label = 'step size $dt=0.29$')
plt.bar(xs+0.25*dx,hist_large,width=0.5*dx, label = 'step size $dt=0.0014$')

plt.plot(hist_teo[:,0],hist_teo[:,1], '--k' , label = 'exact solution')

#plt.xticks((np.pi/4)*np.arange(5), ['$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$','$\pi$'])

plt.xlabel(r"Hitting time $(t_h)$")
plt.ylabel(r"Probability density function")

plt.tight_layout()
plt.legend()

plt.savefig('histogram_hitting.svg', format='svg')
plt.show()
