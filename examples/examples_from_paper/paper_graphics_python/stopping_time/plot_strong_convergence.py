# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

strong_error = np.genfromtxt('./sources/strong_convergence_test.csv', delimiter=',')

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

plt.errorbar(strong_error[:,0],strong_error[:,1],strong_error[:,2],None, 'C0o',label='Euler scheme')
plt.errorbar(strong_error[:,0],strong_error[:,3],strong_error[:,4],None, 'C1o',label='Milstein scheme')
plt.errorbar(strong_error[:,0],strong_error[:,5],strong_error[:,6],None, 'C2o',label='Wagner-Platen scheme')

xs = np.array([strong_error[0,0],0.1])
plt.plot(xs,0.08*xs**(1/4), color='0.5', linestyle='dashed')
plt.plot(xs,0.12*xs**(1/2), color='0.5', linestyle='dashed')
plt.plot(xs,0.30*xs**(1/1), color='0.5', linestyle='dashed')

plt.xlabel(r"Step size")
plt.ylabel(r"Hitting time strong error")

plt.legend()

plt.tight_layout()

plt.savefig('strong_error.svg', format='svg')
plt.show()
