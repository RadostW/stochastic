# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

hist_evensen = np.genfromtxt('./sources/phi_hist_evensen.csv', delimiter=',')
hist_pychastic = np.genfromtxt('./sources/phi_hist_pychastic.csv', delimiter=',')

hist_teo = np.genfromtxt('./sources/phi_hist_teo.csv', delimiter=',')

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

n = len(hist_evensen)
xs = np.linspace(0,np.pi,n)
dx = xs[1]-xs[0]
plt.bar(xs-0.25*dx,hist_pychastic,width=0.5*dx, label = 'Pychastic')
plt.bar(xs+0.25*dx,hist_evensen,width=0.5*dx, label = 'Evensen et al (2008)')

plt.plot(hist_teo[:,0],hist_teo[:,1], '--k' , label = 'SO(3) symmetric')

plt.xticks((np.pi/4)*np.arange(5), ['$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$','$\pi$'])

plt.xlabel(r"Rotation angle $\Phi$")
plt.ylabel(r"Probability density function")

plt.tight_layout()
plt.legend()

plt.savefig('histogram.svg', format='svg')
plt.show()
