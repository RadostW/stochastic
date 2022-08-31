# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

curve_1 = np.genfromtxt('./sources/curve_1.csv', delimiter=',')
curve_2 = np.genfromtxt('./sources/curve_2.csv', delimiter=',')
curve_3 = np.genfromtxt('./sources/curve_3.csv', delimiter=',')

curve_4 = np.genfromtxt('./sources/curve_4.csv', delimiter=',')
curve_5 = np.genfromtxt('./sources/curve_5.csv', delimiter=',')
curve_6 = np.genfromtxt('./sources/curve_6.csv', delimiter=',')

curve_teo = np.genfromtxt('./sources/curve_teo.csv', delimiter=',')

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

plt.plot(curve_1[:,0],curve_1[:,1], 'C0', label = 'Pychastic')
plt.plot(np.insert(curve_2[:,0], 0, 0), np.insert(curve_2[:,1], 0, 0), 'C1', label = 'Evensen et al (2008)')

plt.plot(curve_3[:,0],curve_3[:,1], 'C0')
plt.plot(np.insert(curve_4[:,0], 0, 0), np.insert(curve_4[:,1], 0, 0), 'C1')
plt.plot(curve_5[:,0],curve_5[:,1], 'C0')
plt.plot(np.insert(curve_6[:,0], 0, 0), np.insert(curve_6[:,1], 0, 0), 'C1')

plt.plot(curve_teo[:,0],curve_teo[:,1], '--k' , label = 'Cichocki et al (2015)')

plt.xlim([0.0,2.0])
plt.ylim([0.0,0.28])

plt.xlabel(r"Time $t$ [$\pi \eta d^3 / k_B T$]")
plt.ylabel(r"Modified rotation variance")

plt.tight_layout()
plt.legend()

plt.savefig('correlation_plot.svg', format='svg')
plt.show()
