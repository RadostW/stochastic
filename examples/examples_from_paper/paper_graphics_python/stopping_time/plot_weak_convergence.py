# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

weak_error = np.genfromtxt('./sources/weak_convergence_test.csv', delimiter=',')

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

plt.xscale("log")
plt.yscale("log")

plt.errorbar(weak_error[:,0],weak_error[:,1],weak_error[:,2],None, 'C0o',label='Euler scheme')
plt.errorbar(weak_error[:,0],weak_error[:,3],weak_error[:,4],None, 'C1o',label='Milstein scheme')
plt.errorbar(weak_error[:,0],weak_error[:,5],weak_error[:,6],None, 'C2o',label='Wagner-Platen scheme')

plt.plot(weak_error[:,0],0.1*weak_error[:,0], color='0.5', linestyle='dashed')
plt.plot(weak_error[:,0],0.05*weak_error[:,0]**(3/2), color='0.5', linestyle='dashed')

plt.xlabel(r"Step size")
plt.ylabel(r"Hitting time weak error")

plt.legend()

plt.tight_layout()

plt.savefig('weak_error.svg', format='svg')
plt.show()
