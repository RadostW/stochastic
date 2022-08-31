# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

big_bead_msd = np.genfromtxt('./sources/big_bead_msd.csv', delimiter=',')
centre_msd = np.genfromtxt('./sources/centre_msd.csv', delimiter=',')
small_bead_msd = np.genfromtxt('./sources/small_bead_msd.csv', delimiter=',')
time = np.genfromtxt('./sources/time.csv', delimiter=',')

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


plt.plot(time,small_bead_msd, 'C0', label = 'small bead')
plt.plot(time,big_bead_msd, 'C1',  label = 'large bead')
plt.plot(time,centre_msd, 'C2', label = 'diffusion centre')
plt.plot(time,(0.2898/np.pi)*time, '--k' , label = 'Cichocki et al (2019)')

plt.xlim([0,4000])
plt.ylim([0,550])
plt.xlabel(r"Time [$\tau_d$]")
plt.ylabel(r"Mean square displacement [$a$]")

plt.tight_layout()
plt.legend()

plt.savefig('mean_square_displacement.svg', format='svg')
plt.show()
